#include "graph_filter.h"

namespace token_filter {

std::string GraphFilter::node_filter_dims_to_string() {
    std::stringstream ss;
    for (const auto& nfd : node_filter_dims) {
        ss << nfd.name << std::endl;
        std::queue<FilterDims> dims = nfd.changed_dims;
        while (!dims.empty()) {
            FilterDims dim = dims.front();
            dims.pop();
            ss << dim.index << " " << dim.bsz_index << " " << static_cast<int>(dim.mode) << std::endl;
        }
        ss << "---" << std::endl; // Separator for each node
    }
    return ss.str();
}

// Convert string format to NodeFilterDims (same as load_node_filter_dims)
void GraphFilter::string_to_node_filter_dims(std::string input_str) {
    if (!input_str.empty()){
        std::istringstream iss(input_str);
        std::string line;
        while (std::getline(iss, line)) {
            if (line == "---") {
                continue;
            }
            NodeFilterDims nfd(line);
            while (std::getline(iss, line) && line != "---") {
                std::istringstream line_iss(line);
                int index, bsz_index, mode;
                if (!(line_iss >> index >> bsz_index >> mode)) {
                    throw std::runtime_error("Error reading node filter dimensions from string.");
                }
                auto filter_dims = FilterDims(index, bsz_index, static_cast<FilterDimsMode>(mode));
                nfd.add_changed_dim(filter_dims);
            }
            node_filter_dims.push_back(nfd);
        }
        is_loaded_from_file = true;
    }
}

GraphFilter::GraphFilter(Tensor* mask_) {
    // Get the size of the mask
    IntArrayRef mask_size = mask_->sizes();
    bsz = mask_->sizes()[0];
    seq_before_filter = mask_->sizes()[1];
    seq_after_filter = mask_->sum().item<int>() / bsz;
    
    // Generate mask index
    #ifdef VERBOSE
    std::cout << "Mask size = " << mask_size << std::endl;
    #endif
    
    mask_index_flatten_ = torch::where(mask_->flatten() == 1)[0];
    mask_index_ = (mask_->flatten() == 1).view(mask_size);
    mask_index_seq_ = torch::where(mask_->select(0, 0) == 1)[0];
    
    // Generate mask for re-computing softmax_sum (causal mask)
    attn_mask_softmax = torch::full_symint(
        {seq_after_filter, seq_after_filter}, -std::numeric_limits<float>::infinity(), 
        mask_->options().dtype(torch::kFloat32)
    );
    attn_mask_softmax.triu_(1);

    process_counter = 0;
}

GraphFilter::~GraphFilter() {
    mask_index_flatten_.reset(); 
    mask_index_seq_.reset();
}

Variable GraphFilter::lse_padding(Variable &var, int dim) {
    auto sizes = var.sizes().vec();
    auto pad_size = (kAlignLSE - (sizes[dim] % kAlignLSE)) % kAlignLSE;
    if (pad_size > 0) {
        sizes[dim] += pad_size;
        auto padded_var = torch::zeros(sizes, var.options());
        padded_var.narrow(dim, 0, var.size(dim)).copy_(var);
        var.reset();
        return padded_var;
    }
    return var;
}

void GraphFilter::cache_node_filter_dims(std::string file_path) {
    std::ofstream file(file_path, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        for (const auto& nfd : node_filter_dims) {
            file << nfd.name << std::endl;
            std::queue<FilterDims> dims = nfd.changed_dims;
            while (!dims.empty()) {
                FilterDims dim = dims.front();
                dims.pop();
                file << dim.index << " " << dim.bsz_index << " " << static_cast<int>(dim.mode) << std::endl;
            }
            file << "---" << std::endl; // Separator for each node
        }
        file.close();
    } else {
        throw std::runtime_error("Unable to open file for writing node filter dimensions.");
    }
}

void GraphFilter::load_node_filter_dims(std::string file_path) {
    std::ifstream file(file_path);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (line == "---") {
                continue;
            }
            NodeFilterDims nfd(line);
            while (std::getline(file, line) && line != "---") {
                std::istringstream iss(line);
                int index, bsz_index, mode;
                if (!(iss >> index >> bsz_index >> mode)) {
                    throw std::runtime_error("Error reading node filter dimensions from file.");
                }
                auto filter_dims = FilterDims(index, bsz_index, static_cast<FilterDimsMode>(mode));
                nfd.add_changed_dim(filter_dims);
            }
            node_filter_dims.push_back(nfd);
        }
        file.close();
    } else {
        throw std::runtime_error("Unable to open file for reading node filter dimensions.");
    }
    is_loaded_from_file = true;
}

FilterDims GraphFilter::get_filter_dims_mode(std::vector<c10::SymInt> &sizes) {
    if(is_loaded_from_file) {
        return node_filter_dims[nfd_queue_index].get_changed_dim();
    } else {
        FilterDims filter_dims(-1, -1, FilterDimsMode::NA);
        for(int i=0; i<sizes.size(); i++) {
            if(sizes[i] == seq_before_filter) {
                filter_dims.index = filter_dims.index == -1 ? i : (filter_dims.index + i * 10);
                filter_dims.mode = FilterDimsMode::SEQ_LEN; 
                #ifdef VERBOSE
                std::cout << "get_filter_dims_mode stack the index " << i << " " << filter_dims.index << std::endl;
                if(filter_dims.index >= 10) std::cout << "Get more than one filtering dim" << std::endl;
                #endif
                #ifdef EAGER_BMM_PROCESS
                break;
                #endif
            }
            else if(sizes[i] == seq_before_filter * bsz) {filter_dims.index=i; filter_dims.mode=FilterDimsMode::BTH_SEQ_LEN; break;}
            else if(sizes[i] == seq_before_filter - 1) {filter_dims.index=i; filter_dims.mode=FilterDimsMode::SEQ_LEN_MINUS_1; break;}
            else if(sizes[i] == (seq_before_filter-1)*bsz) {filter_dims.index=i; filter_dims.mode=FilterDimsMode::BTH_SEQ_LEN_MINUS_1; break;}
        }
        for(int i=0; i<sizes.size(); i++) if(sizes[i] == bsz) {filter_dims.bsz_index = i; break;}
        node_filter_dims[nfd_queue_size-1].add_changed_dim(filter_dims);
        return filter_dims;
    }
}

void GraphFilter::process_sizes(std::vector<c10::SymInt> &sizes, int size_index) {
    if (sizes[size_index] == seq_before_filter) sizes[size_index] = seq_after_filter;
    else if (sizes[size_index] == seq_before_filter * bsz) sizes[size_index] = seq_after_filter * bsz;
    else if (sizes[size_index] == seq_before_filter - 1) sizes[size_index] = seq_after_filter - 1;
    else if (sizes[size_index] == (seq_before_filter-1)*bsz) sizes[size_index] = (seq_after_filter-1)*bsz;
}

void GraphFilter::process_sizes(std::vector<c10::SymInt> &sizes) {
    FilterDims filter_dims = get_filter_dims_mode(sizes);
    #ifdef VERBOSE
    std::cout << "process_sizes filter_dims.index=" << filter_dims.index << " filter_dims.mode=" << static_cast<int>(filter_dims.mode) << std::endl;
    #endif
    switch(filter_dims.mode) {
        case FilterDimsMode::SEQ_LEN: 
            if(filter_dims.index < 10) {
                sizes[filter_dims.index] = seq_after_filter; 
            } else {
                sizes[filter_dims.index % 10] = seq_after_filter;
                sizes[filter_dims.index / 10] = seq_after_filter;
            }
            break;
        case FilterDimsMode::BTH_SEQ_LEN: 
            sizes[filter_dims.index] = seq_after_filter * bsz; break;
        case FilterDimsMode::SEQ_LEN_MINUS_1: 
            sizes[filter_dims.index] = seq_after_filter - 1; break;
        case FilterDimsMode::BTH_SEQ_LEN_MINUS_1: 
            sizes[filter_dims.index] = (seq_after_filter - 1) * bsz; break;
    }
}

void GraphFilter::process_sizes(c10::SymInt &size) {
    if(size % seq_before_filter == 0) size = seq_after_filter * (size / seq_before_filter);
}

SavedVariable GraphFilter::process_variable(Variable &var, bool is_output, bool is_lse_padding) {
    #ifdef VERBOSE
    std::cout << "Debug 3" << std::endl;
    #endif

    auto var_sizes = var.sizes();
    std::vector<c10::SymInt> var_size_symint(var_sizes.begin(), var_sizes.end());
    FilterDims filter_dims = get_filter_dims_mode(var_size_symint);
    
    if(map_.find(var.const_data_ptr()) == map_.end()) {
        #ifdef VERBOSE
        std::cout << "process_variable var_sizes=" << var_sizes << " ";
        #endif
        
        Variable new_var; 
        switch (filter_dims.mode) {
            case FilterDimsMode::SEQ_LEN: {
                #ifdef MASKED_SELECT
                new_var = var.masked_select(mask_index_);
                #else
                if(bsz == 1 || filter_dims.bsz_index == -1) {
                    if(filter_dims.index < 10) {
                        new_var = var.index_select(filter_dims.index, mask_index_seq_);
                    } else {
                        Variable tmp = var.index_select(filter_dims.index % 10, mask_index_seq_);
                        new_var = tmp.index_select(filter_dims.index / 10, mask_index_seq_);
                    }
                } else {
                    TORCH_CHECK(filter_dims.index < 10, "Not implemented when filter_dims.index > 10 & bsz > 1");
                    std::vector<c10::SymInt> new_var_size_symint(var_sizes.begin(), var_sizes.end());
                    new_var_size_symint[filter_dims.index] = seq_after_filter;
                    new_var = torch::empty_symint(new_var_size_symint, var.options());
                    
                    std::vector<torch::indexing::TensorIndex> bsz_slice_index(var_sizes.size(), torch::indexing::Slice());
                    for(int i=0; i<bsz; i++) {
                        bsz_slice_index[filter_dims.bsz_index] = i;
                        new_var.index_put_(bsz_slice_index, 
                            var.select(filter_dims.bsz_index, i).index_select(
                            filter_dims.bsz_index < filter_dims.index ? filter_dims.index - 1 : filter_dims.index, 
                            torch::where(mask_index_.select(0, i))[0])
                        ); 
                        #ifdef VERBOSE
                        std::cout << "new_var.index_put_ " << new_var << std::endl;
                        exit(-1);
                        #endif
                    }
                    #ifdef VERBOSE
                    std::cout << "new_var size = " << new_var.sizes() << std::endl;
                    std::cout << "filter.bsz_index = " << filter_dims.bsz_index << std::endl;
                    std::cout << "filter.index = " << filter_dims.index << std::endl;
                    #endif
                }
                #endif
                break;
            } 
            case FilterDimsMode::BTH_SEQ_LEN: {
                #ifdef MASKED_SELECT
                new_var = var.masked_select(mask_index_.view_symint({seq_before_filter * bsz}));
                #else
                new_var = var.index_select(filter_dims.index, mask_index_flatten_);
                #ifdef VERBOSE
                for(int i=0; i<bsz; i++) {
                    std::cout << "batch index = " << i << std::endl;
                    std::cout << "var[i, :100] = " << var.sizes() << std::endl;
                    std::cout << "new_var[i, :100] = " << new_var.sizes() << std::endl;
                }
                #endif
                #endif
                break;
            }
            case FilterDimsMode::SEQ_LEN_MINUS_1: {
                #ifdef MASKED_SELECT
                new_var = var.masked_select(mask_index_.slice_symint(1, 0, seq_before_filter-1));
                #else
                if(bsz == 1 || filter_dims.bsz_index == -1) {
                    new_var = var.index_select(filter_dims.index, mask_index_seq_.slice_symint(0, 0, seq_after_filter-1));
                } else {
                    std::vector<c10::SymInt> new_var_size_symint(var_sizes.begin(), var_sizes.end());
                    new_var_size_symint[filter_dims.index] = seq_after_filter - 1;
                    new_var = torch::empty_symint(new_var_size_symint, var.options());
                    
                    std::vector<torch::indexing::TensorIndex> bsz_slice_index(var_sizes.size(), torch::indexing::Slice());
                    for(int i=0; i<bsz; i++) {
                        bsz_slice_index[filter_dims.bsz_index] = i;
                        new_var.index_put_(bsz_slice_index, 
                            var.select(filter_dims.bsz_index, i).index_select(
                            filter_dims.bsz_index < filter_dims.index ? filter_dims.index - 1 : filter_dims.index, 
                            torch::where(mask_index_.select(0, i).slice_symint(0, 0, seq_before_filter-1))[0])
                        ); 
                        #ifdef VERBOSE
                        std::cout << "batch index = " << i << std::endl;
                        std::cout << "var[i, :100] = " << var.sizes() << std::endl;
                        std::cout << "new_var[i, :100] = " << new_var.sizes() << std::endl;
                        #endif
                    }
                }
                #endif
                break;
            }
            case FilterDimsMode::BTH_SEQ_LEN_MINUS_1: {
                #ifdef MASKED_SELECT
                new_var = var.masked_select(mask_index_.slice_symint(1, 0, seq_before_filter-1));
                #else
                new_var = var.index_select(
                    filter_dims.index, 
                    torch::where(mask_index_.slice_symint(1, 0, seq_before_filter-1).flatten())[0]);
                #ifdef VERBOSE
                for(int i=0; i<bsz; i++) {
                    std::cout << "batch index = " << i << std::endl;
                    std::cout << "var[i, :100] = " << var.sizes() << std::endl;
                    std::cout << "new_var[i, :100] = " << new_var.sizes() << std::endl;
                }
                #endif
                #endif
                break;
            }
        }
        
        if(is_lse_padding) new_var = lse_padding(new_var, filter_dims.index);
        if(!new_var.defined()) new_var = var;
        map_[var.const_data_ptr()] = new_var;
        
        #ifdef VERBOSE
        std::cout << " after=" << map_[var.const_data_ptr()].sizes() << std::endl;
        #endif
        
        return SavedVariable(map_[var.const_data_ptr()], is_output);
    } else {
        auto hist_sizes = map_[var.const_data_ptr()].sizes();
        std::vector<c10::SymInt> new_var_sizes(var_sizes.begin(), var_sizes.end());
        
        switch (filter_dims.mode) {
            case FilterDimsMode::SEQ_LEN: 
                if(filter_dims.index < 10) {
                    new_var_sizes[filter_dims.index] = seq_after_filter; 
                } else {
                    new_var_sizes[filter_dims.index % 10] = seq_after_filter;
                    new_var_sizes[filter_dims.index / 10] = seq_after_filter;
                }
                break;
            case FilterDimsMode::BTH_SEQ_LEN: 
                new_var_sizes[filter_dims.index] = seq_after_filter * bsz; break;
            case FilterDimsMode::SEQ_LEN_MINUS_1: 
                new_var_sizes[filter_dims.index] = seq_after_filter - 1; break;
            case FilterDimsMode::BTH_SEQ_LEN_MINUS_1:
                new_var_sizes[filter_dims.index] = (seq_after_filter - 1) * bsz; break;
        }

        #ifdef VERBOSE
        std::cout << "new_var_sizes=" << new_var_sizes << std::endl;
        std::cout << "hist_sizes=" << hist_sizes << std::endl;
        #endif

        bool is_size_match = true;
        for(int i=0; i<std::min(new_var_sizes.size(), hist_sizes.size()); i++) 
            if(new_var_sizes[i] != hist_sizes[i]) {is_size_match = false; break;}
        
        if(is_size_match) return SavedVariable(map_[var.const_data_ptr()].view_symint(new_var_sizes), is_output);
        else {
            if(
                new_var_sizes.size() == 4 && hist_sizes.size() == 2 &&
                new_var_sizes[0] * new_var_sizes[2] == hist_sizes[0] && 
                new_var_sizes[1] * new_var_sizes[3] == hist_sizes[1]){
                return SavedVariable(
                  map_[var.const_data_ptr()].view_symint({new_var_sizes[0], new_var_sizes[2], new_var_sizes[1], new_var_sizes[3]}).transpose(1, 2), is_output);
              }
              else if(
                new_var_sizes.size() == 2 && hist_sizes.size() == 4 &&
                hist_sizes[0] * hist_sizes[2] == new_var_sizes[0] && 
                hist_sizes[1] * hist_sizes[3] == new_var_sizes[1]){
                return SavedVariable(
                  map_[var.const_data_ptr()].transpose(1, 2).reshape_symint({new_var_sizes[0], new_var_sizes[1]}), is_output);
              }
              else if(
                new_var_sizes.size() == 4 && hist_sizes.size() == 3 &&
                new_var_sizes[0] == hist_sizes[0] &&
                new_var_sizes[2] == hist_sizes[1] &&
                new_var_sizes[1] * new_var_sizes[3] == hist_sizes[2]
              ){
                return SavedVariable(
                    map_[var.const_data_ptr()].view_symint({
                        hist_sizes[0], hist_sizes[1], new_var_sizes[1], new_var_sizes[3]}).transpose(1, 2),
                    is_output
                );
              }
              else if(
                // softmax case
                new_var_sizes.size() == 4 && hist_sizes.size() == 3 &&
                new_var_sizes[0] * new_var_sizes[1] == hist_sizes[0] &&
                new_var_sizes[2] == hist_sizes[1] &&
                new_var_sizes[3] == hist_sizes[2]
              ){
                return SavedVariable(
                  map_[var.const_data_ptr()].view_symint({new_var_sizes[0], new_var_sizes[1], new_var_sizes[2], new_var_sizes[3]}), is_output);
              }
              else{
                std::cout << "new_var_sizes.size(): " << new_var_sizes.size() << " hist_sizes.size(): " << hist_sizes.size() << std::endl;
                std::cout << "new_var_sizes: " << new_var_sizes << " hist_sizes: " << hist_sizes << std::endl;
                TORCH_CHECK(
                  false, 
                  " where the size of the variable is changed. new_var_sizes: ", 
                  new_var_sizes, ", hist_sizes: ", hist_sizes);
              }
        }
    }
}

} // namespace token_filter
