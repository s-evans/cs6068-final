struct node_t {
    unsigned int weight;
    unsigned short left_idx;
    unsigned char symbol;
};

void make_huffman_tree(
        const unsigned int* const d_sorted_histogram,
        const unsigned int* const d_sorted_symbols,
        const unsigned int histogram_size);
