import pandas as pd
from scipy import interpolate
from flexllm.util import base_path


class Swapper:

    """
    A utility class for calculating data swap time between CPU memory and 
    GPU memory by PCIe. It reads pre-measured bandwidth profiles, 
    interpolates bandwidth for specific memory sizes, and computes the 
    time required for swapping model blocks.
    Supports two directions: Host-to-Device (htod) and Device-to-Host (dtoh).
    """

    def __init__(self) -> None:
        htod_path = f"{base_path}/PCIe/htod.csv"
        dtoh_path = f"{base_path}/PCIe/dtoh.csv"

        htod_df = pd.read_csv(htod_path, 
            dtype={"mem": int, "bandwidth(MB/ms)": float})
        dtoh_df = pd.read_csv(dtoh_path, 
            dtype={"mem": int, "bandwidth(MB/ms)": float})
        
        self.htod_df = htod_df.sort_values(by="mem").reset_index(drop=True)
        self.dtoh_df = dtoh_df.sort_values(by="mem").reset_index(drop=True)
    
    def set(self, 
        hidden_size: int, 
        num_layers: int, 
        block_size: int, 
        dtype: int = 2
    ) -> None:
        """
        Set model configuration and calculate block memory and corresponding bandwidth.
        Computes the total memory size of a single model block and interpolates the
        PCIe bandwidth for that memory size from the preloaded data.

        Args:
            hidden_size: Hidden dimension size of the model.
            num_layers: Number of model layers.
            block_size: Token number whose key-value data in stored in the same block.
            dtype: Data type size in bytes (2 for FP16/BF16, 4 for FP32).
        """  
        
        self.block_mem = hidden_size * num_layers * block_size * dtype
        self.htod_bandwidth = self._interpolate_bandwidth(
                df=self.htod_df, 
                target_mem=self.block_mem)
        self.dtoh_bandwidth = self._interpolate_bandwidth(
                df=self.dtoh_df, 
                target_mem=self.block_mem)      

    def _interpolate_bandwidth(self, 
        df: pd.DataFrame, 
        target_mem: int
    ) -> float:
        """
        Interpolate the bandwidth for a given memory size from the measured data.
        First checks for an exact memory match; if not found, performs linear interpolation.
        Raises errors for out-of-bound memory values or interpolation failures.

        Args:
            df: Measured data, conatains 'mem' field and 'bandwidth(MB/ms)' field.
            target_mem: Target memory size in bytes to query bandwidth for.
        
        Returns:
            Interpolated bandwidth value in MB/ms.
        
        Raises:
            ValueError: Target memory is outside the range of measured data.
            RuntimeError: Interpolation process fails.
        """

        mem_match = df[df["mem"] == target_mem]
        if not mem_match.empty:
            return float(mem_match["bandwidth(MB/ms)"].iloc[0])
        
        mem_vals = df["mem"].values
        bw_vals = df["bandwidth(MB/ms)"].values
        min_mem = mem_vals.min()
        max_mem = mem_vals.max()
        if target_mem < min_mem or target_mem > max_mem:
            raise ValueError(f"target memory={target_mem} beyond [{min_mem}, {max_mem}]")
        
        try:
            interp_func = interpolate.interp1d(mem_vals, bw_vals,
                kind="linear", fill_value="extrapolate")
            interpolated_bw = float(interp_func(target_mem))
            return interpolated_bw
        except Exception as e:
            raise RuntimeError(f"interpolate failed: {e}")
    
    def get_swap_time(self, block_num: int, type: str) -> float:
        """
        Calculate the time required to swap a given number of blocks between 
        CPU and GPU. Converts memory size from bytes to MB and computes time 
        using the pre-calculated bandwidth.

        Args:
            block_num: Number of model blocks to swap.
            type: Swap direction, either 'htod' (Host->Device) or 'dtoh' (Device->Host).
        
        Returns:
            Swap time in milliseconds.
        
        Raises:
            Exception: An invalid swap direction type is provided.
        """

        # block_mem(B), self.htod_bandwidth(MB/ms) -> ms
        if type == "htod":
            return block_num * self.block_mem / self.htod_bandwidth / 1024 / 1024
        elif type == "dtoh":
            return block_num * self.block_mem / self.dtoh_bandwidth / 1024 / 1024
        else:
            raise Exception(f"Invalid type: {type}")
        