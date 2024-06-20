import tensorflow as tf
import grpc
import logging
from grpc import RpcError
from proto_files import predict_pb2, prediction_service_pb2
from grpc.beta import implementations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictClient:
    def __init__(self, host, port, model_name, model_version):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_version = model_version

    def _get_signature_key(self, signature):
        if signature in ['serving_default', 'value']:
            return 'value'
        if signature == 'policy':
            return 'policy'
        return 'error'

    def predict(self, input_data, signature_name='serving_default', timeout=10, input_shape=[8*8*13]):
        logger.info(f'Sending request to TF serving model: {self.model_name}, version: {self.model_version}, host: {self.host}')

        tensor_proto = tf.make_tensor_proto(input_data, dtype=tf.float32, shape=input_shape)

        channel = implementations.insecure_channel(self.host, int(self.port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        
        request = predict_pb2.PredictRequest(
            model_spec=predict_pb2.ModelSpec(
                name=self.model_name, 
                signature_name=signature_name, 
                version=tf.make_tensor_proto(self.model_version) if self.model_version > 0 else None
            ),
            inputs={'x': tensor_proto}
        )

        try:
            response = stub.Predict(request, timeout=timeout)
            return list(response.outputs[self._get_signature_key(signature_name)].float_val)
        except RpcError as e:
            logger.error(f'Prediction failed: {e}')

# Example usage
if __name__ == "__main__":
    client = PredictClient('localhost', 8500, 'my_model', 1)
    data = [0.0] * (8 * 8 * 13)
    print(client.predict(data))
