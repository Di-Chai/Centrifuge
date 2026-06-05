#include "common.h"
#include "graph_filter.h"
#include "backward_filters.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace token_filter {

// ========== Main API Function Implementation ==========

std::vector<std::string> collect_node_names(at::Tensor& loss){
    Node* loss_grad_fn = loss.grad_fn().get();
    std::unordered_set<Edge> seen_edges;
    std::unordered_set<std::string> seen_edge_function_names;
    std::queue<Edge> edge_queue;
    for(size_t i=0; i<loss_grad_fn->next_edges().size(); ++i) {
        Edge& edge = loss_grad_fn->next_edges()[i];
        edge_queue.push(edge);
        seen_edges.emplace(edge);
        seen_edge_function_names.emplace(edge.function->name());
    }
    int counter = 0;
    while(!edge_queue.empty()){
        auto current_edge = edge_queue.front();
        edge_queue.pop();
        edge_list& next_edges = current_edge.function->next_edges();
        if(!next_edges.empty())
        for(size_t i = 0; i < next_edges.size(); ++i) {
            Edge& next_edge = next_edges[i];
            if(next_edge.function && seen_edges.emplace(next_edge).second){
                edge_queue.push(next_edge);
                seen_edge_function_names.emplace(next_edge.function->name());
            }
        }
    }
    return std::vector<std::string>(seen_edge_function_names.begin(), seen_edge_function_names.end());
}

void backward_filter_base(at::Tensor& loss, at::Tensor& mask, GraphFilter& graph_filter){
    // Get all the edges
    Node* loss_grad_fn = loss.grad_fn().get();
    std::unordered_set<Edge> seen_edges;
    std::unordered_set<std::string> seen_edge_function_names;
    std::queue<Edge> edge_queue;
    std::vector<Edge*> all_edges;
    for(size_t i=0; i<loss_grad_fn->next_edges().size(); ++i) {
        Edge& edge = loss_grad_fn->next_edges()[i];
        edge_queue.push(edge);
        seen_edges.emplace(edge);
        seen_edge_function_names.emplace(edge.function->name());
        #ifdef VERBOSE
        std::cout << "edge.function->name(): " << edge.function->name() << std::endl;
        #endif
    }
    int counter = 0;
    while(!edge_queue.empty()){
        auto current_edge = edge_queue.front();
        edge_queue.pop();
        edge_list& next_edges = current_edge.function->next_edges();
        if(!next_edges.empty())
        for(size_t i = 0; i < next_edges.size(); ++i) {
            Edge& next_edge = next_edges[i];
            if(next_edge.function && seen_edges.emplace(next_edge).second){
                edge_queue.push(next_edge);
                seen_edge_function_names.emplace(next_edge.function->name());
                all_edges.push_back(&next_edge);
            }
        }
    }
    
    // Process all the edges
    at::Tensor attn_bias_restoreq; // util variables
    for (size_t i = 0; i < all_edges.size(); ++i) {
        #ifdef TIME_EVAL
        // Time measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        #endif
        Edge* edge = all_edges[i];
        auto fn = edge->function.get();
        
        // Process current function
        if(!graph_filter.is_loaded_from_file) 
        graph_filter.add_node_filter_dims(fn->name());
        #ifdef EAGER_BMM_PROCESS
        if(fn->name() != "BmmBackward0") 
        node_processing(fn, &graph_filter);
        #else
        node_processing(fn, &graph_filter);
        #endif
        graph_filter.nfd_queue_index++;

        // Specific function processing (BMM)
        #ifdef EAGER_BMM_PROCESS
        if(edge.function->name() == "BmmBackward0") 
            process_bmm_backward(edge, &mask);
        #endif

        // Specific function processing (Attention)
        #ifdef RESTORE_Q
        bool is_attn_edge = false;
        if(fn->name() == "ScaledDotProductFlashAttentionBackward0") {
            process_scaled_dot_product_flash_attn_restore_q(edge, &mask, attn_bias_restoreq);
            is_attn_edge = true;
        }
        if(fn->name() == "ScaledDotProductEfficientAttentionBackward0") {
            process_scaled_dot_product_efficient_attn(edge, &mask, attn_bias_restoreq);
            is_attn_edge = true;
        }
        if(fn->name() == "ScaledDotProductCudnnAttentionBackward0") {
            process_scaled_dot_product_cudnn_attn(edge, &mask);
            is_attn_edge = true;
        }
        if(is_attn_edge){
            if(!graph_filter.is_loaded_from_file) graph_filter.add_node_filter_dims(edge->function->name());
            node_processing(edge->function.get(), &graph_filter);
            graph_filter.nfd_queue_index++;
        }
        #endif
        #ifdef TIME_EVAL
        // Time measurement
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << edge->function->name() << " Time taken: " << duration.count() << " microseconds" << std::endl;
        #endif
    }
}

// New backward_filter function that accepts and returns node_filter_dims as string
std::string backward_filter_with_dims(at::Tensor loss, at::Tensor mask, std::string input_node_filter_dims_str) {

    #ifdef TIME_EVAL
    // Time measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif
    
    GraphFilter graph_filter(&mask);  

    // Load node_filter_dims from input if provided
    if (!input_node_filter_dims_str.empty())
    graph_filter.string_to_node_filter_dims(input_node_filter_dims_str);

    backward_filter_base(loss, mask, graph_filter);
    
    #ifdef TIME_EVAL
    // Time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken by backward_filter_with_dims: " << duration.count() << " milliseconds" << std::endl;
    #endif

    // Return the node_filter_dims as string
    return graph_filter.node_filter_dims_to_string();
}

at::Tensor backward_filter(at::Tensor loss, at::Tensor mask) {

    #ifdef TIME_EVAL
    // Time measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif

    // Launch parallel subprocesses for backward_filter_parallel
    #ifdef PARALLEL_FILTER
    const int num_threads = 1;  // Configurable number of threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            backward_filter_parallel_v2(loss, mask, i, num_threads); // 
        });
    }
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    #else

    // Util variables
    at::Tensor attn_bias_restoreq;
    GraphFilter graph_filter(&mask);  
    // Node tracing file
    std::string node_tracing_file = "/tmp/node_tracing.txt";
    if (std::ifstream(node_tracing_file)) 
    graph_filter.load_node_filter_dims(node_tracing_file);
    
    Node* loss_grad_fn = loss.grad_fn().get();
    std::unordered_set<Node*> seen_nodes;
    std::unordered_set<std::string> seen_node_names;
    std::queue<Node*> next_functions;
    
    next_functions.push(loss_grad_fn);
    while(!next_functions.empty()) {
        // Get front function
        auto fn = next_functions.front();
        next_functions.pop();
        
        // Obtain next edges
        if(seen_nodes.emplace(fn).second) {
            // Check the fn.name()
            seen_node_names.emplace(fn->name());
            
            // Process current function
            if(!graph_filter.is_loaded_from_file) graph_filter.add_node_filter_dims(fn->name());
            
            #ifdef EAGER_BMM_PROCESS
            if(fn->name() != "BmmBackward0") node_processing(fn, &graph_filter);
            #else
            node_processing(fn, &graph_filter);
            #endif
            
            graph_filter.nfd_queue_index++;
            
            // Push the edges
            edge_list& edges = fn->next_edges();
            #ifdef VERBOSE
            std::cout << "edges.size(): " << edges.size() << std::endl;
            #endif
            
            if(!edges.empty())
            for(size_t i = 0; i < edges.size(); ++i) {
                auto& edge = edges[i];
                if(edge.function) {
                    #ifdef VERBOSE
                    std::cout << "edge.function->name(): " << edge.function->name() << std::endl;
                    #endif
                    
                    // Process BmmBackward
                    #ifdef EAGER_BMM_PROCESS
                    if(edge.function->name() == "BmmBackward0") 
                        process_bmm_backward(edge, &mask);
                    #endif
                    
                    // Process ScaledDotProductFlashAttentionBackward0
                    #ifdef RESTORE_Q_NAIVE
                    if(edge.function->name() == "ScaledDotProductFlashAttentionBackward0") 
                        process_scaled_dot_product_flash_attn(edge, &mask);
                    #endif
                    
                    #ifdef RESTORE_Q
                    if(edge.function->name() == "ScaledDotProductFlashAttentionBackward0") {
                        if(!graph_filter.is_loaded_from_file) graph_filter.add_node_filter_dims(edge.function->name());
                        node_processing(edge.function.get(), &graph_filter);
                        graph_filter.nfd_queue_index++;
                        process_scaled_dot_product_flash_attn_restore_q(edge, &mask, attn_bias_restoreq);
                        graph_filter.process_counter -= 1;
                    }
                    if(edge.function->name() == "ScaledDotProductEfficientAttentionBackward0") {
                        if(!graph_filter.is_loaded_from_file) graph_filter.add_node_filter_dims(edge.function->name());
                        node_processing(edge.function.get(), &graph_filter);
                        graph_filter.nfd_queue_index++;
                        process_scaled_dot_product_efficient_attn(edge, &mask, attn_bias_restoreq);
                    }
                    if(edge.function->name() == "ScaledDotProductCudnnAttentionBackward0") {
                        if(!graph_filter.is_loaded_from_file) graph_filter.add_node_filter_dims(edge.function->name());
                        node_processing(edge.function.get(), &graph_filter);
                        graph_filter.nfd_queue_index++;
                        process_scaled_dot_product_cudnn_attn(edge, &mask);
                    }
                    #endif
                    
                    next_functions.push(edge.function.get());
                }
            }
        }
    }

    if(!std::ifstream(node_tracing_file)) 
    graph_filter.cache_node_filter_dims(node_tracing_file);

    #endif // PARALLEL_FILTER

    #ifdef TIME_EVAL
    // Time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken by backward_filter: " << duration.count() << " milliseconds" << std::endl;
    #endif

    return loss;
}

} // namespace token_filter

// ========== PyBind11 Bindings ==========

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(token_filter, m) {
    m.def("backward_filter(Tensor loss, Tensor mask) -> Tensor");
    m.def("backward_filter_with_dims(Tensor loss, Tensor mask, str input_node_filter_dims) -> str");
    m.def("collect_node_names(Tensor loss) -> str[]");
}

TORCH_LIBRARY_IMPL(token_filter, CPU, m) {
    m.impl("backward_filter", &token_filter::backward_filter);
    m.impl("backward_filter_with_dims", &token_filter::backward_filter_with_dims);
    m.impl("collect_node_names", &token_filter::collect_node_names);
}

TORCH_LIBRARY_IMPL(token_filter, CUDA, m) {
    m.impl("backward_filter", &token_filter::backward_filter);
    m.impl("backward_filter_with_dims", &token_filter::backward_filter_with_dims);
    m.impl("collect_node_names", &token_filter::collect_node_names);
}
