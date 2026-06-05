import socket
import argparse
import torch
import time
import pickle
import multiprocessing

from functools import partial
from model.utils import *
from model.n_gram import NGramModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def handle_client(client_id, client_socket, queue):
    """Handles communication with a single client."""
    try:
        while True:
            message = receive_all(client_socket)
            if not message:
                break
            messages = pickle.loads(message)
            queue.put((client_id, messages))
    except ConnectionResetError:
        print("Client disconnected abruptly.")
    finally:
        client_socket.close()

def start_server(host, port, model_name, target_tokenizer=None, context_length=2048, model_type='llm'):
    if model_type == 'llm':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # build the device map
        # config = AutoConfig.from_pretrained(model_name)
        # num_gpus = torch.cuda.device_count()
        # num_first_state_layers = 1
        # num_last_state_layers = 9
        # num_middle_state_layers = (config.num_hidden_layers - num_first_state_layers - num_last_state_layers) // (num_gpus-2)
        # layer_split = [num_first_state_layers] + [num_middle_state_layers] * (num_gpus-2) + [num_last_state_layers]
        # device_map = {
        #     'model.embed_tokens': 0,
        #     'lm_head': 0, 
        #     'model.norm': num_gpus-1, 
        #     'model.rotary_emb': num_gpus-1,
        # }
        # layer_counter = 0
        # for i in range(num_gpus):
        #     for j in range(layer_split[i]):
        #         device_map[f'model.layers.{layer_counter}'] = i
        #         layer_counter += 1

        # Load the model and tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            # device_map=device_map,
        ).to(device)
        model.eval()

        tokenize_args = {
            "truncation": True, "max_length": context_length, 
            "return_overflowing_tokens": False, "return_length": False, 
            "padding_side": "left"
        }

        if target_tokenizer is not None:
            target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
            if target_tokenizer.pad_token_id is None:
                target_tokenizer.pad_token = target_tokenizer.eos_token

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        obtain_reference_loss = partial(
            obtain_ref_loss, model=model, tokenizer=tokenizer, 
            target_tokenizer=target_tokenizer, tokenize_args=tokenize_args
        )

    elif model_type in ('bigram', 'trigram'):
        model = NGramModel(model_name)
        obtain_reference_loss = partial(
            model.obtain_ngram_loss, model=model_type
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    """Starts the server and listens for incoming client connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    client_sockets = []
    while len(client_sockets) < args.num_connects:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        client_sockets.append(client_socket)
    
    # Start process for each client connection
    recv_q = multiprocessing.Queue()
    for c_id in range(len(client_sockets)):
        client_process = multiprocessing.Process(
            target=handle_client, args=(c_id, client_sockets[c_id], recv_q)
        )
        client_process.daemon = True
        client_process.start()

    try:
        sample_counter = 0
        start = time.time()
        while True:
            # Wait for a message from the queue
            client_id, messages = recv_q.get()
            assert len(messages) > 0
            # Process the message and obtain the loss
            # response = obtain_ref_loss(model, tokenizer, messages, target_tokenizer, tokenize_args)
            response = obtain_reference_loss(messages=messages)
            # print("Server processes message", client_id, [len(e) for e in messages], "Costs", time.time()-st)
            # Send the result back to the corresponding client
            send_all(client_sockets[client_id], response)
            sample_counter += len(messages)
            print(f"\rServer processes {(sample_counter / (time.time() - start)):.2f} samples/second", flush=True, end="")
    except Exception as e:
        print("Server shutting down.", e)
    finally:
        server_socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Start a server for handling client requests.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP address')
    parser.add_argument('--port', type=int, default=65432, help='Host port number')
    parser.add_argument('--num_connects', type=int, default=4, help='Number of connections to accept')
    parser.add_argument('--model_type', type=str, choices=('llm', 'bigram', 'trigram'), default='llm', help='Model type')
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='Model name to load')
    parser.add_argument('--target_tokenizer', type=str, default=None, help='Target tokenizer to be aligned')
    parser.add_argument('--context_length', type=int, default=2048, help='Maximum context length')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_server(
        host=args.host, port=args.port, model_name=args.model_name, 
        target_tokenizer=args.target_tokenizer, context_length=args.context_length,
        model_type=args.model_type
    )