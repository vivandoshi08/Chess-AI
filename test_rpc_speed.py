import sys
import time
import chess
import pychess_utils as chess_utils
from rpc_client import PredictClient

sys.path.append(sys.path[0] + "/..")

def run_test(client, num_predictions):
    start_time = time.time()
    for _ in range(num_predictions):
        client.predict(chess_utils.expand_position(chess.Board()), signature_name='policy')
    serial_duration = time.time() - start_time
    print(f"{num_predictions} predictions done separately took: {serial_duration:.4f} seconds")

    start_time = time.time()
    client.predict([chess_utils.expand_position(chess.Board()) for _ in range(num_predictions)], 
                   request_timeout=num_predictions, signature_name='policy', shape=[num_predictions, 832])
    batch_duration = time.time() - start_time
    print(f"{num_predictions} predictions done in one batch took: {batch_duration:.4f} seconds")

    print(f"Per prediction times: {serial_duration / num_predictions:.4f} vs. {batch_duration / num_predictions:.4f}")
    print(f"Savings per prediction: {(serial_duration / num_predictions) - (batch_duration / num_predictions):.4f}")

def main():
    client = PredictClient('127.0.0.1', 9000, 'default', int(chess_utils.latest_version()))
    print("Performing one dummy prediction...")
    client.predict(chess_utils.expand_position(chess.Board()))
    print("Starting test")

    for size in (2 ** x for x in range(8)):
        run_test(client, size)

if __name__ == "__main__":
    main()
