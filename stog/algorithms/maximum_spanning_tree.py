import numpy as np

def decode(energies, lengths, num_leading_symbols=0, labeled=True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 4D tensor
        energies of each edge. the shape is [batch_size, num_labels, n_steps, n_steps],
        where the dummy root is at index 0.
    :param masks: numpy 2D tensor
        masks in the shape [batch_size, n_steps].
    :param num_leading_symbols: int
        number of symbolic dependency types leading in type alphabets)
    :return:
    """

    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(1, length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        id = 0
        for cyc_node in cycle:
            cyc_nodes[id] = cyc_node
            id += 1
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in range(cyc_len):
                j1 = cyc_nodes[j]
                if score_matrix[j1, i] > max1:
                    max1 = score_matrix[j1, i]
                    wh1 = j1

                scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

                if scr > max2:
                    max2 = scr
                    wh2 = j1

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for i in range(1, cyc_len):
            cyc_node = cyc_nodes[i]
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    if labeled:
        assert energies.ndim == 4, 'dimension of energies is not equal to 4'
    else:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    for i in range(batch_size):
        energy = energies[i]

        # calc the realy length of this instance
        length = lengths[i]

        # calc real energy matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic types).
        if labeled:
            energy = energy[num_leading_symbols:, :length, :length]
            # get best label for each edge.
            label_id_matrix = energy.argmax(axis=0) + num_leading_symbols
            energy = energy.max(axis=0)
        else:
            energy = energy[:length, :length]
            label_id_matrix = None
        # get original score matrix
        orig_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = np.array(orig_score_matrix, copy=True)

        oldI = np.zeros([length, length], dtype=np.int32)
        oldO = np.zeros([length, length], dtype=np.int32)
        curr_nodes = np.zeros([length], dtype=np.bool)
        reps = []

        for s in range(length):
            orig_score_matrix[s, s] = 0.0
            score_matrix[s, s] = 0.0
            curr_nodes[s] = True
            reps.append(set())
            reps[s].add(s)
            for t in range(s + 1, length):
                oldI[s, t] = s
                oldO[s, t] = t

                oldI[t, s] = t
                oldO[t, s] = s

        final_edges = dict()
        chuLiuEdmonds()
        par = np.zeros([max_length], np.int32)
        if labeled:
            type = np.ones([max_length], np.int32)
            type[0] = 0
        else:
            type = None

        for ch, pr in final_edges.items():
            par[ch] = pr
            if labeled and ch != 0:
                type[ch] = label_id_matrix[pr, ch]

        par[0] = 0
        pars[i] = par
        if labeled:
            types[i] = type

    return pars, types