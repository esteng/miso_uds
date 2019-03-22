import argparse
from collections import Counter
from stog.data.dataset_readers.semantic_dependency_parsing \
    import SemanticDependenciesDatasetReader

def print_line():
    print("=" * 89)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('extract_amr_token.py')
    parser.add_argument('file_path')
    parser.add_argument('--show_stats', action='store_true')
    args = parser.parse_args()

    dataset_reader = SemanticDependenciesDatasetReader()

    original_relation_counter = Counter()
    generate_relation_counter = Counter()

    num_instance = 0
    num_instance_with_isolated_edges = 0
    num_tgt_token_list = []
    num_edge_list = []
    num_src_token_list = []
    num_isolated_edge_list = []
    num_generated_edge_list = []
    num_target_copy_list = []

    for instance in dataset_reader._read(args.file_path):
        tgt_tokens = [
            x.text for x in instance.fields["tgt_tokens"].tokens[1:-1]
        ]
        num_tgt_token_list.append(len(tgt_tokens))

        src_tokens = [
            x.text for x in instance.fields["src_tokens"].tokens[1:-1]
        ]
        num_src_token_list.append(len(src_tokens))

        original_edges = instance.fields["arc_indices"].metadata
        num_edge_list.append(len(original_edges))

        isolated_edges = instance.fields["isolated_edges"].metadata
        num_isolated_edge_list.append(len(isolated_edges))

        if len(isolated_edges) > 0:
            num_instance_with_isolated_edges += 1

        original_relations = instance.fields["arc_tags"].metadata
        for relation in original_relations:
            original_relation_counter[relation] += 1

        generate_relations = instance.fields["head_tags"].labels
        for relation in generate_relations:
            generate_relation_counter[relation] += 1
        
        num_target_copy_list.append(
            sum(instance.fields["tgt_copy_mask"].labels)
        )

        num_instance += 1

    print_line()
    print("Number of instance (None empty): {}".format(num_instance))

    print(
        "Number of instance with isolated edges: {}, ({:.2f} %)".format(
            num_instance_with_isolated_edges,
            float(100 * num_instance_with_isolated_edges / num_instance)
        )
    )
    print("Number of annotated edges: {}".format(sum(num_edge_list)))

    print(
        "Number of isolated edges: {}, ({:.2f} %)".format(
            sum(num_isolated_edge_list),
            float(100 * sum(num_isolated_edge_list) / sum(num_edge_list))
        )
    )
    print("10 Most frequet original labels")
    for k, v in original_relation_counter.most_common(10):
        print(
            "\t{}: {} ({:.2f} %)".format(
                k, v, 100 * v / sum(original_relation_counter.values())
            )
        )
    print_line()
    print(
        "Number of generated relations: {}".format(
            sum(generate_relation_counter.values())
        )
    )
    print("10 Most frequet generated labels")
    for k, v in generate_relation_counter.most_common(10):
        print(
            "\t{}: {} ({:.2f} %)".format(
                k, v, 100 * v / sum(generate_relation_counter.values())
            )
        )
    print_line()
    print(
        "Number of reversed relations: {}, ({:.2f} %)".format(
            sum(
                [
                    v for k, v in generate_relation_counter.items()
                    if "_reversed" in k
                ]
            ),
            100 * sum(
                [
                    v for k, v in generate_relation_counter.items()
                    if "_reversed" in k
                ]
            ) / sum(generate_relation_counter.values())

        )
    )
    print("Details on each reversed relation")
    for k, v in generate_relation_counter.most_common():
        if "_reversed" in k:
            num_total = v
            original_relation = k.replace("_reversed", "")
            if original_relation in generate_relation_counter:
                num_total += generate_relation_counter[original_relation]
            print(
                "\t{}: {} / {} ({:.2f} %)".format(
                    original_relation,
                    v,
                    num_total,
                    v / num_total * 100
                )
            )

    print_line()
    print(
        "Number of copied relations: {}, ({:.2f} %)".format(
            sum(num_target_copy_list),
            100
            * sum(num_target_copy_list)
            / sum(generate_relation_counter.values())
        )
    )
    print_line()
