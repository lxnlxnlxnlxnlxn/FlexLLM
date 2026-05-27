import os
from typing import Union, List
import pandas as pd
import joblib
from flexllm.util import base_path


class Predictor:
    
    """
    A model inference time predictor for FlexLLM. It loads pre-trained 
    models to predict prefill time and decode time based on batch size 
    and sequence length. Supports both single and batch inputs for 
    time prediction.
    """

    def __init__(self, model_name: str) -> None:
        predictor_path = f"{base_path}/forward_time/{model_name}"
        assert os.path.isdir(predictor_path), f"Invalid model: {model_name} !"
        self.model_name = model_name

        # Load prefill time prediction model
        ptime_path = predictor_path + "/prefill.pkl"
        assert os.path.exists(ptime_path), f"prefill time unprocessed !"
        self.ptime_predictor = joblib.load(ptime_path)

        # Load decode time prediction model
        dtime_path = predictor_path + "/decode.pkl"
        assert os.path.exists(dtime_path), f"decode time unprocessed !"
        self.dtime_predictor = joblib.load(dtime_path)

    def get_time(self, 
        bs: Union[int, List[int]], 
        seql: Union[int, List[int]],
        stage: str
    ) -> Union[int, List[int]]:
        """
        Predict model inference time for given inputs. Supports both single 
        value inputs and list inputs for batch prediction.
        
        Args:
            bs: Batch size (single int or list of batch sizes).
            seql: Sequence length (single int or list of sequence lengths).
            stage: Inference stage, 'p' for prefill, 'd' for decode.
        
        Returns:
            Predicted time in milliseconds (single float or list of floats).
        
        Raises:
            Exception: Invalid inference stage is provided.
            RuntimeError: Prediction process fails.
        """
        
        is_single_input = False
        if isinstance(bs, int):
            is_single_input = True
            bs = [bs]
            seql = [seql]
        
        if stage == "p":
            model = self.ptime_predictor
        elif stage == "d":
            model = self.dtime_predictor
        else:
            raise Exception(f"Invalid stage: {stage}")
        
        try:
            X_pred = pd.DataFrame({'bs': bs, 'seql': seql})
            pred_times = model.predict(X_pred)
            pred_times = [float(t) for t in pred_times]
        except Exception as e:
            raise RuntimeError(f"Get an failure in prediction: {e}")

        return pred_times[0] if is_single_input else pred_times
    