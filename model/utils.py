import time
import json
import torch
import socket
import pickle
import multiprocessing
import numpy as np

from torch.cuda import nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

class BatchTimer:
    def __init__(self, gradient_acc_steps, output_file, warmup=2):
        self.batch_start = None
        self.batch_end = None
        self.batch_time = {}
        self.all_batches = []
        self._grad_acc_steps = gradient_acc_steps
        self._last_record = None
        self._output_file = output_file
        self._record_names = {"total"}
        self._nvtx_profile = True
        self._step = 0
        self._warmup = warmup

    def start(self):
        self._step += 1
        self.batch_start = time.time()
        self._last_record = self.batch_start
        if self._nvtx_profile:
            if self._step == 1:
                torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push(f"Step {self._step} start")

    def end(self):
        assert self.batch_start is not None, "Batch timer not started"
        self.batch_end = time.time()
        self.batch_time["total"] = self.batch_end - self.batch_start
        self.all_batches.append(self.batch_time)
        self.batch_time = {}
        self.batch_start = None
        self._last_record = None
        self.dump()
        if self._nvtx_profile and self._step > 5:
            torch.cuda.cudart().cudaProfilerStop()

    def record(self, name):
        assert self._last_record is not None, "Batch timer not started"
        cur_time = time.time()
        self.batch_time[name] = self.batch_time.get(name, 0) + cur_time - self._last_record
        self._last_record = cur_time
        self._record_names.add(name)
    
    def record_nvtx(self, name):
        if self._nvtx_profile:
            nvtx.range_pop()
            nvtx.range_push(name)

    def last_batch(self):
        return self.all_batches[-1]
    
    def dump(self):
        averaged_time = {}
        for name in self._record_names:
            counter = 0
            time_sum = 0
            for batch in self.all_batches[self._warmup:]:
                if name in batch:
                    counter += 1
                    time_sum += batch[name]
            averaged_time[name + "_avg"] = time_sum / (1 if counter == 0 else counter)
        with open(self._output_file, 'w') as f:
            json.dump(self.all_batches + [averaged_time], f, indent=4)

@torch.inference_mode()
def obtain_ref_loss(model, tokenizer, messages, target_tokenizer=None, 
                    tokenize_args={}, return_pickle=True, manuel_matching=True):
    if manuel_matching and target_tokenizer is not None:
        target_input_ids = messages
        messages = target_tokenizer.batch_decode(
            (messages) if not target_tokenizer.add_bos_token else ([e[1:] for e in messages]))
        inputs = tokenizer(messages, return_tensors="pt", padding=True, **tokenize_args).to('cuda' if torch.cuda.is_available() else 'cpu')
        # Forward pass through the model
        outputs = model(**inputs)
        # t1 = time.time()
        # Forward pass through the model
        outputs = model(**inputs)
        # Calculate the loss for the batch
        loss = causal_modeling_loss(
            logits=outputs.logits, inputs=inputs.input_ids, attention_mask=inputs.attention_mask, average=False)
        
        size_of_token = [sum(inputs.attention_mask[e].detach().cpu().tolist()) for e in range(len(inputs.attention_mask))]
        loss_list = loss.detach().cpu().tolist()
        loss_list = [loss_list[e][-(size_of_token[e]-1):] for e in range(len(loss_list))]
        
        # Align the loss using the target tokenizer
        if target_tokenizer is not None:
            ref_input_ids = inputs.input_ids.detach().cpu().tolist()
            ref_input_ids = [ref_input_ids[e][-(size_of_token[e]):] for e in range(len(ref_input_ids))]
            ref_tokens = tokenized_prompts(tokenizer, input_ids=ref_input_ids)
            target_tokens = tokenized_prompts(target_tokenizer, input_ids=target_input_ids)
            loss_list = align_ref_target_tokens(
                ref_tokens=ref_tokens,
                target_tokens=target_tokens,
                ref_loss=loss_list
            )
    else:
        msg_size = [len(e) for e in messages]
        if len(set(msg_size)) != 1:
            max_msg_size = max(msg_size)
            messages = [([tokenizer.pad_token_id] * (max_msg_size-len(e)) + e) for e in messages]
        input_ids = torch.asarray(messages).to('cuda' if torch.cuda.is_available() else 'cpu')
        # Forward pass through the model
        outputs = model(input_ids=input_ids)
        loss = causal_modeling_loss(
            logits=outputs.logits, inputs=input_ids, attention_mask=None, average=False)
        loss_list = loss.detach().cpu().tolist()
        loss_list = [loss_list[e][-(msg_size[e]-1):] for e in range(len(loss_list))]
    
    # Serialize the loss object to a byte stream
    if return_pickle:
        response = pickle.dumps({
            "loss": loss_list
        })
        return response
    else:
        return loss_list

def receive_all(client_socket):
    """Receives all data from the client."""
    # First, receive the size of the data
    data_size = int.from_bytes(client_socket.recv(4), byteorder='big')
    data = b""
    while len(data) < data_size:
        part = client_socket.recv(min(4096, data_size - len(data)))
        data += part
    return data

def send_all(client_socket, data):
    """Sends all data to the client."""
    # First, send the size of the data
    data_size = len(data)
    client_socket.sendall(data_size.to_bytes(4, byteorder='big'))
    # Then, send all the data
    client_socket.sendall(data)

def init_recv_queue(socket, queue):
    while True:
        try:
            data = receive_all(socket)
            queue.put(pickle.loads(data))
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

def init_send_queue(socket, queue):
    while True:
        try:
            data = queue.get()
            send_all(socket, pickle.dumps(data))
            time.sleep(0.1)
        except Exception as e:
            print(f"Error sending data: {e}")
            break

def init_single_socket(host, port):
    print("Connecting to", host, port)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, int(port)))
    # Create a message queue
    recv_queue = multiprocessing.Queue()
    send_queue = multiprocessing.Queue()
    # Start a process to handle async receiving of data
    receive_process = multiprocessing.Process(target=init_recv_queue, args=(client_socket, recv_queue))
    receive_process.daemon = True
    receive_process.start()
    # Start a process to handle async sending of data
    send_process = multiprocessing.Process(target=init_send_queue, args=(client_socket, send_queue))
    send_process.daemon = True
    send_process.start()
    return recv_queue, send_queue

def init_multiple_sockets(hosts, ports):
    """
    For pre-computing reference loss
    """
    hosts = hosts.split("@")
    ports = [e.split(",") for e in ports.split("@")]
    assert len(hosts) == len(ports), "Invalid hosts:ports configuration."
    queues = []
    for i in range(len(hosts)):
        for j in range(len(ports[i])):
            queues.append(init_single_socket(hosts[i], ports[i][j]))
    return queues

def init_socket(args, selected_host=None, return_all_queues=False):
    """
    Todo: need to be updated
    """
    hosts = args.ref_socket_hosts.split(",")
    ports = args.ref_socket_ports.split(",")
    assert len(hosts) == 1 or len(hosts) == len(ports), "Invalid hosts:ports configuration."
    if torch.distributed.is_initialized():
        selected_host = selected_host or torch.distributed.get_rank() % len(ports)
    else:
        selected_host = selected_host or 0
    if len(hosts) > 1:
        host = hosts[selected_host]
    else:
        host = hosts[0]
    port = int(ports[selected_host])
    if not return_all_queues:
        return init_single_socket(host, port)
    else:
        return [init_socket(args, i, False) for i in range(len(ports))]

# def init_hf_processor(args, recv_queue, send_queue, device='cuda:0'):
#     ref_model = AutoModelForCausalLM.from_pretrained(
#             args.ref_model_path, torch_dtype=torch.bfloat16).to(device).eval()
#     ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
#     # Target tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     # Align the tokenizer
#     ref_tokenizer.add_bos_token = tokenizer.add_bos_token
#     if ref_tokenizer.pad_token is None:
#         ref_tokenizer.pad_token = ref_tokenizer.eos_token
#     ref_batch_size = 32 # decide base on GPU memory
#     # Process the received tokens
#     while True:
#         try:
#             message = send_queue.get()
#             ref_loss = []
#             for i in range(0, len(message), ref_batch_size):
#                 ref_loss += obtain_ref_loss(
#                     model=ref_model, tokenizer=ref_tokenizer,
#                     messages=message[i:i+ref_batch_size],
#                     target_tokenizer=tokenizer, return_pickle=False, manuel_matching=False,
#                     tokenize_args={
#                         "max_length": args.context_length, "truncation": True,
#                         "return_overflowing_tokens": False, "return_length": False
#                     }
#                 )
#             recv_queue.put(ref_loss)
#         except Exception as e:
#             print("Processor shutting down.", e)
#             break

# def init_hf_recv_send_queue(args, device='cuda:0'):
#     # Create a message queue
#     recv_queue = multiprocessing.Queue()
#     send_queue = multiprocessing.Queue()
#     # Start hf process to process the reference model loss
#     hf_process = multiprocessing.Process(
#         target=init_hf_processor, args=(args, recv_queue, send_queue, device)
#     )
#     hf_process.daemon = True
#     hf_process.start()
#     return recv_queue, send_queue

def is_left_padding(attention_mask):
    with torch.no_grad():
        left_mask_col = attention_mask[:, 0].sum().item()
        right_mask_col = attention_mask[:, -1].sum().item()
        if left_mask_col <= right_mask_col:
            return True
        else:
            return False

def collate_fn(batch):
    longest = max([len(x['input_ids']) for x in batch])
    result = {}
    # left padding
    new_input_ids = np.stack([[0] * (longest-len(x["input_ids"])) + x["input_ids"] for x in batch])
    if 'attention_mask' not in batch[0]:
        for i in range(len(batch)):
            batch[i]["attention_mask"] = [1] * len(batch[i]["input_ids"])
    new_attention_mask = np.stack([[0] * (longest-len(x["attention_mask"])) + x["attention_mask"] for x in batch])
    result["input_ids"] = torch.from_numpy(new_input_ids)
    result["attention_mask"] = torch.from_numpy(new_attention_mask)
    if "text" in batch[0]:
        result["text"] = [x["text"] for x in batch]
    if "ref_loss" in batch[0]:
        result["ref_loss"] = [x["ref_loss"] for x in batch]
    if "length" in batch[0]:
        result["length"] = [x["length"] for x in batch]
    return result

def apply_mask_to_loss(loss, attention_mask):
    if attention_mask is None:
        return loss
    else:
        if is_left_padding(attention_mask):
            return loss * attention_mask[..., 1:]

def loss_average(loss, attention_mask, by_sample=False, is_left_padding=None):
    if attention_mask is not None:
        is_left_padding = is_left_padding or is_left_padding(attention_mask)
        if is_left_padding:
            loss_mask = attention_mask[..., :-1]
        else:
            loss_mask = attention_mask[..., 1:]
        loss = torch.multiply(loss, loss_mask)
        loss = loss.sum(-1) / loss_mask.sum(-1)
    else:
        loss = loss.mean(-1)
    if by_sample:
        return loss
    else:
        return loss.mean()

def token_filter_loss(
        inputs, logits, attention_mask=None, ref_loss=None, dropping_strategy=None, 
        drop_rate=None, is_left_padding=None, return_mask=False, batch_timer=None):
    if batch_timer is not None:
        batch_timer.record_nvtx("causal_modeling_loss")
    loss = causal_modeling_loss(inputs, logits, attention_mask=None, average=False)
    if batch_timer is not None:
        batch_timer.record("causal_modeling_loss")
    is_left_padding = is_left_padding or is_left_padding(attention_mask)
    if ref_loss is not None:
        # padding the ref_loss
        if batch_timer is not None:
            batch_timer.record_nvtx("token_dropping_mask")
        for i in range(len(ref_loss)):
            ref_loss[i] = [0.0] * (len(loss[i])-len(ref_loss[i])) + ref_loss[i]
        ref_mask = token_dropping_mask(
            loss, torch.asarray(ref_loss).to(inputs.device), attention_mask, 
            strategy=dropping_strategy, drop_rate=drop_rate,
            left_padding=is_left_padding
        )
        # ref_mask = attention_mask
        # ref_mask[:, :int(attention_mask.shape[1] * drop_rate)] = 0
        # time.sleep(0.1)
        if batch_timer is not None:
            batch_timer.record("token_dropping_mask")
        if not return_mask:
            return loss_average(loss, ref_mask, by_sample=False, is_left_padding=is_left_padding)
        else:
            return loss_average(loss, ref_mask, by_sample=False, is_left_padding=is_left_padding), ref_mask
    else:
        if not return_mask:
            return loss_average(loss, attention_mask, by_sample=False, is_left_padding=is_left_padding)
        else:  
            return loss_average(loss, attention_mask, by_sample=False, is_left_padding=is_left_padding), attention_mask

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    counter = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            logits = model(batch["input_ids"]).logits
            loss = token_filter_loss(batch["input_ids"], logits, attention_mask=batch["attention_mask"], is_left_padding=True)
        counter += batch["input_ids"].size(0)
        losses.append(accelerator.gather(loss))
    accelerator.wait_for_everyone()
    loss = torch.mean(torch.cat(losses, 0))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def evaluate_single_gpu(model, eval_dataloader, device="cuda"):
    torch.cuda.empty_cache()
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            try:
                batch_input_ids = batch["input_ids"].to(model.device)
            except:
                batch_input_ids = batch["input_ids"].to(device)
            logits = model(batch_input_ids).logits
            loss = token_filter_loss(batch_input_ids, logits, attention_mask=batch["attention_mask"].to(device), is_left_padding=True)
        losses.append(loss)
    loss = torch.mean(torch.hstack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    torch.cuda.empty_cache()
    return loss.item(), perplexity.item()

def causal_modeling_loss(inputs, logits, attention_mask=None, is_left_padding=None, average=False, by_sample=False):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size(0), shift_labels.size(1))
    if not average:
        return loss
    else:
        return loss_average(loss, attention_mask, by_sample=by_sample, is_left_padding=is_left_padding)

def token_dropping_mask(loss, ref_loss, attention_mask, strategy="fixed", drop_rate=None, left_padding=None):
    
    def align_drop_rate(drop_rate, align_div=8):
        seq_len = loss.shape[1]
        filter_len = int(seq_len * drop_rate)
        if filter_len % align_div != 0:
            filter_len = filter_len + (align_div - (filter_len % align_div))
            return (filter_len / seq_len) + (0.5 / seq_len)
        else:
            return drop_rate
    
    if strategy == "fixed":
        # attention_mask[:, :int(attention_mask.shape[1] * drop_rate)] = 0
        # process each sample in the batch
        with torch.no_grad():
            loss_diff = loss - ref_loss
            drop_rate = align_drop_rate(drop_rate)
            # determine the padding direction
            left_padding = left_padding or is_left_padding(attention_mask)
            for i in range(loss.shape[0]):
                loss_token_size = attention_mask[i].sum().item() - 1 # loss has one token less than attention_mask
                loss_token_start = int(attention_mask.shape[1] - loss_token_size - 1) if left_padding else 1
                loss_token_end = (attention_mask.shape[1] - 1) if left_padding else int(loss_token_size+1)
                k = int(drop_rate * loss_token_size) # number of loss tokens to drop
                _, indices = torch.topk(loss_diff[i][loss_token_start:loss_token_end], k, largest=False)
                attention_mask[i].scatter_(0, indices+loss_token_start, 0)
        return attention_mask
    elif strategy == "positive":
        with torch.no_grad():
            loss_diff = loss - ref_loss
            # update the drop rate to be the minimum ratio of negative diffs among all samples in the batch
            negative_rate = (loss_diff < 0).sum(1).min() / len(loss_diff[0])
            drop_rate = min(negative_rate, drop_rate)
            drop_rate = max(0.2, drop_rate)
            drop_rate = align_drop_rate(drop_rate)
            with open(f"tmp/drop_rate_{torch.distributed.get_rank()}.txt", "a") as f:
                f.write(f"drop_rate: {drop_rate}\n")
            # determine the padding direction
            left_padding = left_padding or is_left_padding(attention_mask)
            for i in range(loss.shape[0]):
                loss_token_size = attention_mask[i].sum().item() - 1 # loss has one token less than attention_mask
                loss_token_start = int(attention_mask.shape[1] - loss_token_size - 1) if left_padding else 1
                loss_token_end = (attention_mask.shape[1] - 1) if left_padding else int(loss_token_size+1)
                k = int(drop_rate * loss_token_size) # number of loss tokens to drop
                _, indices = torch.topk(loss_diff[i][loss_token_start:loss_token_end], k, largest=False)
                attention_mask[i].scatter_(0, indices+loss_token_start, 0)
        return attention_mask
    elif strategy == "dynamic":
        return NotImplementedError(f"Strategy {strategy} is not implemented.")
    elif strategy == "self":
        with torch.no_grad():
            loss_diff = loss
            # determine the padding direction
            left_padding = left_padding or is_left_padding(attention_mask)
            for i in range(loss.shape[0]):
                loss_token_size = attention_mask[i].sum().item() - 1 # loss has one token less than attention_mask
                loss_token_start = int(attention_mask.shape[1] - loss_token_size - 1) if left_padding else 1
                loss_token_end = (attention_mask.shape[1] - 1) if left_padding else int(loss_token_size+1)
                k = int(drop_rate * loss_token_size) # number of loss tokens to drop
                _, indices = torch.topk(loss_diff[i][loss_token_start:loss_token_end], k, largest=False)
                attention_mask[i].scatter_(0, indices+loss_token_start, 0)
        return attention_mask
    elif strategy == "less_ref":
        with torch.no_grad():
            loss_diff = -1 * ref_loss
            # determine the padding direction
            left_padding = left_padding or is_left_padding(attention_mask)
            for i in range(loss.shape[0]):
                loss_token_size = attention_mask[i].sum().item() - 1 # loss has one token less than attention_mask
                loss_token_start = int(attention_mask.shape[1] - loss_token_size - 1) if left_padding else 1
                loss_token_end = (attention_mask.shape[1] - 1) if left_padding else int(loss_token_size+1)
                k = int(drop_rate * loss_token_size) # number of loss tokens to drop
                _, indices = torch.topk(loss_diff[i][loss_token_start:loss_token_end], k, largest=False)
                attention_mask[i].scatter_(0, indices+loss_token_start, 0)
        return attention_mask
    elif strategy == "peaking":
        loss_diff = loss - ref_loss
        # determine the padding direction
        left_padding = left_padding or is_left_padding(attention_mask)
        with torch.no_grad():
            for i in range(loss.shape[0]):
                loss_token_size = attention_mask[i].sum().item() - 1 # loss has one token less than attention_mask
                loss_token_start = int(attention_mask.shape[1] - loss_token_size - 1) if left_padding else 1
                loss_token_end = (attention_mask.shape[1] - 1) if left_padding else int(loss_token_size+1)

                loss_diff_i = loss_diff[i][loss_token_start:loss_token_end]

                top_40 = loss_diff_i.sort(descending=False)[0][int(drop_rate * loss_token_size)] # number of loss tokens to drop
                top_60 = loss_diff_i.sort(descending=False)[0][int(0.6 * loss_token_size)] # number of loss tokens to keep

                anchor_mask = (loss_diff_i < top_40)

                # print("archored mask ratio = ", float(len(anchor_mask[anchor_mask==True]))/len(anchor_mask))
                cluster_mask = (loss_diff_i < top_60)

                cluster_mask_int = cluster_mask.int()

                padded_mask = torch.cat([torch.tensor([0], device=cluster_mask.device), cluster_mask_int])
                diff = padded_mask[1:] - padded_mask[:-1]

                run_starts = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
                run_ends = (diff == -1).nonzero(as_tuple=False).squeeze(-1)

                if len(run_starts) > len(run_ends):
                    run_ends = torch.cat([run_ends, torch.tensor([len(loss_diff_i)], device=run_ends.device, dtype=torch.long)])

                cluster_lengths = run_ends - run_starts
                min_cluster_length = 3  # Minimum length of a cluster to be considered
                long_enough_mask = (cluster_lengths >= min_cluster_length)
                
                valid_starts = run_starts[long_enough_mask]
                valid_ends = run_ends[long_enough_mask]

                final_selection_mask = torch.zeros_like(cluster_mask, dtype=torch.bool, device=cluster_mask.device)

                for start, end in zip(valid_starts, valid_ends):
                    is_anchored = torch.any(anchor_mask[start:end])
                    if is_anchored:
                        final_selection_mask[start:end] = True
                        
                # --- Step 6: final selection mask ---
                # Indices from the final boolean mask
                
                final_indece = final_selection_mask.nonzero(as_tuple=False).squeeze(-1)
                if long_enough_mask.sum() == 0:
                    final_indece = anchor_mask.nonzero(as_tuple=False).squeeze(-1)
                # print("final_indece = ", final_indece)
                # print("final_indece ration = ", float(len(final_indece))/len(loss_diff_i))
                attention_mask[i].scatter_(0, final_indece+loss_token_start, 0)
        return attention_mask
    else:
        raise NotImplementedError(f"Strategy {strategy} is not supported.")

def tokenized_prompts(tokenizer, prompts=None, tokenizer_args: dict = {}, input_ids=None):
    assert input_ids is not None or prompts is not None, "Either prompts or inputs should be provided."
    if input_ids is None:
        input_ids = tokenizer(prompts, **tokenizer_args)['input_ids']
    return [
        [tokenizer.decode(input_ids[batch_id][i]).strip(" ") 
        for i in range(len(input_ids[batch_id]))]
        for batch_id in range(len(input_ids))
    ]

def update_token_index(ref_token_list, str_index, token_index):
    pre_length, ti = token_index
    while (pre_length + len(ref_token_list[ti])) < str_index:
        pre_length += len(ref_token_list[ti])
        ti += 1
    return [pre_length, ti]

def align_ref_target_tokens(ref_tokens, target_tokens, ref_loss):    
    aligned_losses = []
    for k in range(len(ref_tokens)):
        # t1 = time.time()
        total_operations = 0
        ref_tokens_str = "".join(ref_tokens[k])
        target_tokens_str = "".join(target_tokens[k])
        if ref_tokens_str != target_tokens_str:
            print("Missmacth, aligning the tokens...")
            import pdb; pdb.set_trace()
            ref_tokens_str_index = 0
            target_tokens_str_index = 0
            ref_token_index = [-1, 0]
            while True:
                # Find the first mismatch
                while ref_tokens_str_index<len(ref_tokens_str) and \
                target_tokens_str_index<len(target_tokens_str) and \
                ref_tokens_str[ref_tokens_str_index] == target_tokens_str[target_tokens_str_index]:
                    ref_tokens_str_index += 1
                    target_tokens_str_index += 1
                
                # Chech whether the unprocessed ref or target string is empty
                if ref_tokens_str_index >= len(ref_tokens_str) or target_tokens_str_index >= len(target_tokens_str):
                    if ref_tokens_str_index < len(ref_tokens_str):
                        # Delete the remaining tokens
                        del_ref_length = len(ref_tokens_str) - ref_tokens_str_index
                        deleted_len = 0
                        while deleted_len < del_ref_length:
                            if (del_ref_length - deleted_len) >= len(ref_tokens[k][-1]):
                                deleted_len += len(ref_tokens[k].pop(-1))
                                ref_loss[k].pop(-1)
                            else:
                                ref_tokens[k][-1] = ref_tokens[k][-1][:-(del_ref_length-deleted_len)]
                                deleted_len = del_ref_length
                    elif target_tokens_str_index < len(target_tokens_str):
                        # Insert the remaining tokens
                        ref_tokens[k][-1] += target_tokens_str[target_tokens_str_index:]
                    break
                
                # Compute the edit distance
                process_size = min(32, len(ref_tokens_str)-ref_tokens_str_index, len(target_tokens_str)-target_tokens_str_index)
                edit_distance = np.zeros((process_size + 1, process_size + 1))
                for i in range(edit_distance.shape[0]):
                    for j in range(edit_distance.shape[1]):
                        if i == 0:
                            edit_distance[i][j] = j
                        elif j == 0:
                            edit_distance[i][j] = i
                        elif ref_tokens_str[ref_tokens_str_index:][i - 1] == target_tokens_str[target_tokens_str_index:][j - 1]:
                            edit_distance[i][j] = edit_distance[i - 1][j - 1]
                        else:
                            edit_distance[i][j] = 1 + min(edit_distance[i - 1][j], edit_distance[i][j - 1], edit_distance[i - 1][j - 1])
                # Get the edit operations
                edit_operations = []
                i, j = edit_distance.shape[0]-1, edit_distance.shape[1]-1
                while i > 0 and j > 0:
                    if ref_tokens_str[ref_tokens_str_index:][i - 1] == target_tokens_str[target_tokens_str_index:][j - 1]:
                        edit_operations.append([1, i-1, j-1])  # unchange
                        i -= 1
                        j -= 1
                    elif edit_distance[i][j] == edit_distance[i - 1][j - 1] + 1:
                        edit_operations.append([2, i-1, j-1])  # substitute
                        i -= 1
                        j -= 1
                    elif edit_distance[i][j] == edit_distance[i - 1][j] + 1:
                        edit_operations.append([3, i-1, j-1])  # delete
                        i -= 1
                    elif edit_distance[i][j] == edit_distance[i][j - 1] + 1:
                        edit_operations.append([4, i-1, j-1])  # insert
                        j -= 1
                while i > 0:
                    edit_operations.append([3, i-1, j-1])  # delete
                    i -= 1
                while j > 0:
                    edit_operations.append([4, i-1, j-1])  # insert
                    j -= 1
                # edit_operations.reverse()
                # edit_operations = edit_operations[::-1]

                edit_distance_reversed = []
                for i in range(len(edit_operations)-1, -1, -1):
                    edit_distance_reversed.append(edit_operations[i])
                edit_operations = edit_distance_reversed

                num_del = 0
                num_insert = 0
                # Remove the last insert and delete
                l = len(edit_operations) - 1
                while edit_operations[l][0] == 3 or edit_operations[l][0] == 4:
                    l -= 1
                edit_operations = edit_operations[:l+1]
                
                for i in range(len(edit_operations)):
                    op, _, jj = edit_operations[i]
                    if op == 2:
                        # Find the token index
                        ref_token_index = update_token_index(ref_tokens[k], ref_tokens_str_index+i, ref_token_index)
                        # Get the token and update
                        ref_token_tmp = ref_tokens[k][ref_token_index[1]]
                        ref_token_tmp = ref_token_tmp[:ref_tokens_str_index+i-ref_token_index[0]-1] + target_tokens_str[target_tokens_str_index+jj] + ref_token_tmp[ref_tokens_str_index+i-ref_token_index[0]:]
                        # Update the token list and cooresponding string
                        ref_tokens[k][ref_token_index[1]] = ref_token_tmp
                        ref_tokens_str = ref_tokens_str[:ref_tokens_str_index+i] + target_tokens_str[target_tokens_str_index+jj] + ref_tokens_str[ref_tokens_str_index+i+1:]
                    if op == 3:
                        num_del += 1
                        # Find the token index
                        ref_token_index = update_token_index(ref_tokens[k], ref_tokens_str_index+i, ref_token_index)
                        # Get the token
                        ref_token_tmp = ref_tokens[k][ref_token_index[1]]
                        # Update the token list and cooresponding string
                        ref_tokens[k][ref_token_index[1]] = ref_token_tmp[:ref_tokens_str_index+i-ref_token_index[0]-1] + ref_token_tmp[ref_tokens_str_index+i-ref_token_index[0]:]
                        ref_tokens_str = ref_tokens_str[:ref_tokens_str_index+i] + ref_tokens_str[ref_tokens_str_index+i+1:]
                        ref_tokens_str_index -= 1
                    if op == 4:
                        num_insert += 1
                        # Find the token index
                        ref_token_index = update_token_index(ref_tokens[k], ref_tokens_str_index+i, ref_token_index)
                        # Get the token and update
                        ref_token_tmp = ref_tokens[k][ref_token_index[1]]
                        ref_token_tmp = ref_token_tmp[:ref_tokens_str_index+i-ref_token_index[0]-1] + target_tokens_str[target_tokens_str_index+jj] + ref_token_tmp[ref_tokens_str_index+i-ref_token_index[0]-1:]
                        # Update the token list and cooresponding string
                        ref_tokens[k][ref_token_index[1]] = ref_token_tmp
                        ref_tokens_str = ref_tokens_str[:ref_tokens_str_index+i] + target_tokens_str[target_tokens_str_index+jj] + ref_tokens_str[ref_tokens_str_index+i:]

                # Accumulate the index
                ref_tokens_str_index += (process_size)
                target_tokens_str_index += (process_size - num_del)

                total_operations = total_operations + num_del + num_insert

                # Debug information
                # print(ref_tokens_str[ref_tokens_str_index:ref_tokens_str_index+10], target_tokens_str[target_tokens_str_index:target_tokens_str_index+10])
                # print(ref_tokens_str[:ref_tokens_str_index] == target_tokens_str[:target_tokens_str_index])
                # print("Mathching Process")

        # assert len(ref_loss[k]) == (len(ref_tokens[k]) - 1), f"{len(ref_loss[k])} != {len(ref_tokens[k])-1}"
        # assert "".join(ref_tokens[k]) == "".join(target_tokens[k]), f"Ref and target tokens are not aligned!"
        
        # t2 = time.time(); print("Align strings costs", t2-t1)

        # Loss start at the second token
        ref_tokens[k][1] = ref_tokens[k][0] + ref_tokens[k][1]
        target_tokens[k][1] = target_tokens[k][0] + target_tokens[k][1]
        ref_tokens[k].pop(0)
        target_tokens[k].pop(0)
        
        aligned_loss: list[float] = []
        i = 0; j = 0
        tmp_ref_str = ""
        tmp_target_str = ""
        tmp_target_str_start = 0
        ref_loss_accum = 0.0
        while i < len(ref_tokens[k]) or j < len(target_tokens[k]):
            if tmp_ref_str == "" and tmp_target_str == "":
                if i >= len(ref_tokens[k]) or j >= len(target_tokens[k]):
                    break
                tmp_ref_str = ref_tokens[k][i]
                tmp_target_str = target_tokens[k][j]
                tmp_target_str_start = j
                ref_loss_accum = ref_loss[k][i]
                i += 1
                j += 1
            if len(tmp_ref_str) < len(tmp_target_str):
                ref_loss_accum += ref_loss[k][i]
                tmp_ref_str += ref_tokens[k][i]
                i += 1
            if len(tmp_ref_str) > len(tmp_target_str):
                tmp_target_str += target_tokens[k][j]
                j += 1
            if tmp_ref_str == tmp_target_str:
                for _ in range(j-tmp_target_str_start):
                    aligned_loss.append(ref_loss_accum / (j-tmp_target_str_start))
                # aligned_loss += [ref_loss_accum/(j-tmp_target_str_start)]*(j-tmp_target_str_start)
                tmp_ref_str = ""
                tmp_target_str = ""
                ref_loss_accum = 0.0
        if j < len(target_tokens[k]):
            for _ in range(len(target_tokens[k]) - j):
                aligned_loss.append(0.0)
            # aligned_loss += [0.0] * (len(target_tokens[k]) - j)
        # assert len(aligned_loss) == len(target_tokens[k])
        aligned_losses.append(aligned_loss)
        
        # t3 = time.time(); print("Align tokens costs", t3-t2)

        # print('Total Operations', total_operations)
    
    return aligned_losses
