


def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        call in collate_fn():
            y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
        where:
            item_info["generate_y"] # generate_y: ['cheap', 'none', 'none', 'none', 'none', 'none', 'none', '4', 'yes', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']

        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        # print ("in collate_fn(): lengths: {}".format(lengths))
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        # print ("in collate_fn(): padded_seqs: {}\n".format(padded_seqs))
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

if __name__ == '__main__':
    sequence = 
    merge_multi_response()
