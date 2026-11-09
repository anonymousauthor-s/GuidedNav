"""Chat-boundary parsing shared by the Stage 1/2 dataset wrappers.

This module has no tensor dependency so malformed multi-turn conversations can
be checked without loading the model or a GPU runtime.
"""


def assistant_target_spans(token_ids, *, im_start_id, im_end_id, role_headers):
    """Return [start,end) assistant answer spans, including EOS only.

    role_headers maps user/system/assistant to their complete tokenized
    ``<|im_start|>role\n`` sequence. A word such as "assistant" inside a user
    message is never treated as a role header. Message headers, user content,
    trailing formatting newlines and padding receive no target labels.
    """
    ids = list(token_ids)
    spans, position = [], 0
    while position < len(ids):
        if ids[position] != im_start_id:
            position += 1
            continue
        role, header_length = None, None
        for name, header in role_headers.items():
            header = list(header)
            if ids[position:position + len(header)] == header:
                role, header_length = name, len(header)
                break
        if role is None:
            raise ValueError("Unknown or malformed chat role header")
        answer_start = position + header_length
        end = answer_start
        while end < len(ids) and ids[end] != im_end_id:
            if ids[end] == im_start_id:
                raise ValueError("A new chat message started before the previous message ended")
            end += 1
        if end == len(ids):
            raise ValueError("A training chat message is missing its end token")
        if role == "assistant":
            if end == answer_start:
                raise ValueError("An assistant training answer must not be empty")
            spans.append((answer_start, end + 1))
        position = end + 1
    if not spans:
        raise ValueError("No complete assistant answer found in this training row")
    return spans
