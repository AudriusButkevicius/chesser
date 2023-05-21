import io
import math
import os.path
import pathlib
import time
from typing import Optional

import numpy as np
from toolz import itertoolz


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


BUFFER_SIZE = 4 << 20


def split(input_stream, out_dir: pathlib.Path, items_per_split, prefix):
    paths = []
    current_file_index = 0
    items_so_far = items_per_split
    output = None

    for record in input_stream:
        if items_so_far >= items_per_split:
            items_so_far = 0
            if output:
                output.close()
            current_file_index += 1
            print(f"Starting chunk {current_file_index}")
            chunk_path = out_dir / f"{prefix}-{current_file_index}.bin"
            paths.append(chunk_path)
            output = io.open(chunk_path, "wb", buffering=BUFFER_SIZE)
        output.write(record)
        items_so_far += 1
    return paths


def to_iter(path: pathlib.Path, record_size: int):
    assert os.path.getsize(path) % record_size == 0
    records_per_buffer = BUFFER_SIZE // record_size
    with io.open(path, "rb", buffering=BUFFER_SIZE) as fd:
        multiple_records = fd.read(records_per_buffer * record_size)
        while multiple_records:
            count = len(multiple_records) // record_size
            for i in range(count):
                yield multiple_records[i * record_size : (i + 1) * record_size]
            multiple_records = fd.read(records_per_buffer * record_size)


def to_file(iterator, path: pathlib.Path):
    with io.open(path, "wb", buffering=BUFFER_SIZE) as fd:
        for record in iterator:
            fd.write(record)


def sort(input_file: pathlib.Path, dtype, sort_field: Optional[str] = None):
    print(f"Sorting {input_file}")
    data_array = np.fromfile(input_file, dtype=dtype)
    print(f"Records {len(data_array)}")
    if sort_field:
        sorted_data_array = np.sort(data_array, order=[sort_field], axis=0)
    else:
        sorted_data_array = np.sort(data_array, axis=0)
    sorted_data_array.tofile(input_file)


def shuffle(input_file: pathlib.Path, record_size: int, output_file: Optional[pathlib.Path] = None):
    print(f"Shuffling {input_file}")
    data_array = np.fromfile(input_file, dtype=np.dtype(f"({record_size})b,"))
    print(f"Records {len(data_array)}")
    np.random.shuffle(data_array)
    data_array.tofile(output_file or input_file)


def unique(seq, key):
    previous = None
    dropped = 0
    emitted = 0
    total = 0
    last_print = time.time()
    for item in seq:
        total += 1
        now = time.time()
        if now > last_print + 2:
            last_print = now
            print(
                f"Stats. Emitted: {emitted} ({float(emitted * 100) / total:.2f}%), Dropped: {dropped} ({float(dropped * 100) / total:.2f}%)"
            )
        val = key(item)
        if val == previous:
            dropped += 1
            continue

        previous = val
        yield item

        emitted += 1
    print(f"Stats. Emitted: {emitted} ({float(emitted * 100) / total:.2f}%), Dropped: {dropped} ({float(dropped * 100) / total:.2f}%)")


def main():
    RECORD_SIZE = 92
    UNIQUENESS_SIZE = 7 * 8 + 1 + 2 + 2 + 1 + 8 + 8
    import sys

    print(sys.argv)

    input_file = pathlib.Path(sys.argv[1])

    assert os.path.getsize(input_file) % RECORD_SIZE == 0
    output_path = pathlib.Path(sys.argv[2])

    print(f"Will parse {input_file} into {output_path}")

    os.makedirs(output_path, exist_ok=True)

    desired_chunk_size = 200 << 20
    records_per_split = desired_chunk_size // RECORD_SIZE

    sorted_paths = list(output_path.rglob("sorted*.bin"))
    if not sorted_paths:
        starting_stream = to_iter(input_file, RECORD_SIZE)
        print("Splitting for sorting")
        sorted_paths = split(starting_stream, output_path, records_per_split, "sorted")
        print(f"Splitting for sorting produced {len(sorted_paths)}")

        dtype = np.dtype(f"({UNIQUENESS_SIZE})b, ({RECORD_SIZE-UNIQUENESS_SIZE})b")
        for path in sorted_paths:
            sort(path, dtype, "f0")

        print("Sort done")
    else:
        print("Already sorted")

    print("Splitting for shuffle")
    unique_paths = list(output_path.rglob("unique*.bin"))
    if not unique_paths:
        merged = itertoolz.merge_sorted(*[to_iter(path, RECORD_SIZE) for path in sorted_paths])

        merged_unique = unique(merged, lambda x: x[:UNIQUENESS_SIZE])

        unique_paths = split(merged_unique, output_path, records_per_split, "unique")
    else:
        print("Already shuffle split")

    print(f"Splitting for shuffle produced {len(unique_paths)}")
    for path in unique_paths:
        new_path = pathlib.Path(str(path).replace("unique", "shuffled"))
        shuffle(path, RECORD_SIZE, new_path)


if __name__ == "__main__":
    main()
