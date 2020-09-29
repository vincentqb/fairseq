import copy
from typing import Any, Dict, Iterator, List

import torch

from fairseq import search
from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def build_generator(
    target_dictionary, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
):
    if getattr(args, "score_reference", False):
        from fairseq.sequence_scorer import SequenceScorer

        return SequenceScorer(
            target_dictionary,
            compute_alignment=getattr(args, "print_alignment", False),
        )

    # Choose search strategy. Defaults to Beam Search.
    sampling = getattr(args, "sampling", False)
    sampling_topk = getattr(args, "sampling_topk", -1)
    sampling_topp = getattr(args, "sampling_topp", -1.0)
    diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    match_source_len = getattr(args, "match_source_len", False)
    diversity_rate = getattr(args, "diversity_rate", -1)
    constrained = getattr(args, "constraints", False)
    if (
        sum(
            int(cond)
            for cond in [
                sampling,
                diverse_beam_groups > 0,
                match_source_len,
                diversity_rate > 0,
            ]
        )
        > 1
    ):
        raise ValueError("Provided Search parameters are mutually exclusive.")
    assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

    if sampling:
        search_strategy = search.Sampling(
            target_dictionary, sampling_topk, sampling_topp
        )
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(
            target_dictionary, diverse_beam_groups, diverse_beam_strength
        )
    elif match_source_len:
        # this is useful for tagging applications where the output
        # length should match the input length, so we hardcode the
        # length constraints for simplicity
        search_strategy = search.LengthConstrainedBeamSearch(
            target_dictionary,
            min_len_a=1,
            min_len_b=0,
            max_len_a=1,
            max_len_b=0,
        )
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(
            target_dictionary, diversity_rate
        )
    elif constrained:
        search_strategy = search.LexicallyConstrainedBeamSearch(
            target_dictionary, args.constraints
        )
    else:
        search_strategy = search.BeamSearch(target_dictionary)

    if seq_gen_cls is None:
        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = SequenceGenerator
    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
    return seq_gen_cls(
        models,
        target_dictionary,
        beam_size=getattr(args, "beam", 5),
        max_len_a=getattr(args, "max_len_a", 0),
        max_len_b=getattr(args, "max_len_b", 200),
        min_len=getattr(args, "min_len", 1),
        normalize_scores=(not getattr(args, "unnormalized", False)),
        len_penalty=getattr(args, "lenpen", 1),
        unk_penalty=getattr(args, "unkpen", 0),
        temperature=getattr(args, "temperature", 1.0),
        match_source_len=getattr(args, "match_source_len", False),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        search_strategy=search_strategy,
        **extra_gen_cls_kwargs,
    )


def inference_step(generator, models, sample, prefix_tokens=None, constraints=None):
    with torch.no_grad():
        return generator.generate(
            models, sample, prefix_tokens=prefix_tokens, constraints=constraints
        )


def _build_batches(
    args, task, models, tokens: List[List[int]], skip_invalid_size_inputs: bool
) -> Iterator[Dict[str, Any]]:

    max_positions = resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    lengths = torch.LongTensor([t.numel() for t in tokens])
    batch_iterator = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=skip_invalid_size_inputs,
        disable_iterator_cache=True,
    ).next_epoch_itr(shuffle=False)
    return batch_iterator


def generate(
    args,
    task,
    models,
    device,
    target_dictionary,
    tokenized_sentences: List[torch.LongTensor],
    beam: int = 5,
    verbose: bool = False,
    skip_invalid_size_inputs=False,
    inference_step_args=None,
    **kwargs
) -> List[List[Dict[str, torch.Tensor]]]:
    if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
        return generate(
            tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
        )[0]

    # build generator using current args as well as any kwargs
    gen_args = copy.copy(args)
    gen_args.beam = beam
    for k, v in kwargs.items():
        setattr(gen_args, k, v)
    generator = build_generator(target_dictionary, models, gen_args)

    inference_step_args = inference_step_args or {}
    results = []
    for batch in _build_batches(
        args, task, models, tokenized_sentences, skip_invalid_size_inputs
    ):
        batch = apply_to_sample(lambda t: t.to(device), batch)
        translations = inference_step(generator, models, batch, **inference_step_args)
        for id, hypos in zip(batch["id"].tolist(), translations):
            results.append((id, hypos))

    # sort output to match input order
    outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

    return outputs
