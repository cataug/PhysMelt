#!/usr/bin/env python3

import struct
from pathlib import Path


ROOT = Path.home() / "PhysMelt"
VIDEO = ROOT / "data/raw/NIST_RHF/MPM_AVIs/RHF_MPM_P01.avi"
OUT_DIR = ROOT / "outputs/qc"

WIDTH = 120
HEIGHT = 120
CHANNELS = 3
ROW_STRIDE = ((WIDTH * CHANNELS + 3) // 4) * 4
FRAME_BYTES = ROW_STRIDE * HEIGHT


def iter_chunks(handle, start, end):
    position = start

    while position + 8 <= end:
        handle.seek(position)

        chunk_id = handle.read(4)
        size_raw = handle.read(4)

        if len(chunk_id) != 4 or len(size_raw) != 4:
            break

        size = struct.unpack("<I", size_raw)[0]
        data_start = position + 8
        data_end = min(data_start + size, end)

        yield chunk_id, size, data_start, data_end

        next_position = data_start + size + (size % 2)

        if next_position <= position:
            break

        position = next_position


def find_movi(handle, file_size):
    for chunk_id, size, data_start, data_end in iter_chunks(
        handle, 12, file_size
    ):
        if chunk_id != b"LIST":
            continue

        handle.seek(data_start)
        list_type = handle.read(4)

        if list_type == b"movi":
            return data_start + 4, data_end

    raise RuntimeError("LIST movi section was not found")


def collect_frames(handle, start, end, output):
    for chunk_id, size, data_start, data_end in iter_chunks(
        handle, start, end
    ):
        if chunk_id == b"LIST":
            handle.seek(data_start)
            list_type = handle.read(4)

            if list_type in {b"rec ", b"movi"}:
                collect_frames(
                    handle,
                    data_start + 4,
                    data_end,
                    output,
                )

        elif len(chunk_id) == 4 and chunk_id[2:4] in {b"db", b"dc"}:
            output.append({
                "chunk_id": chunk_id,
                "offset": data_start,
                "size": size,
            })


def decode_dib_frame(handle, frame_info):
    handle.seek(frame_info["offset"])
    payload = handle.read(frame_info["size"])

    if len(payload) < FRAME_BYTES:
        raise RuntimeError(
            f"Frame is too short: {len(payload)} < {FRAME_BYTES}"
        )

    rows = []
    unequal_pixels = 0

    for y in range(HEIGHT):
        row_start = y * ROW_STRIDE
        row = payload[row_start:row_start + WIDTH * CHANNELS]

        gray_row = bytearray(WIDTH)

        for x in range(WIDTH):
            base = x * 3

            blue = row[base]
            green = row[base + 1]
            red = row[base + 2]

            if not (blue == green == red):
                unequal_pixels += 1

            gray_row[x] = round(
                (int(blue) + int(green) + int(red)) / 3
            )

        rows.append(bytes(gray_row))

    # Positive DIB height means rows are stored bottom-up.
    rows.reverse()
    gray = b"".join(rows)

    mismatch_fraction = unequal_pixels / (WIDTH * HEIGHT)

    return gray, mismatch_fraction


def save_pgm(path, image):
    header = f"P5\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(image)


def image_statistics(image):
    values = list(image)

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main():
    print("Video:", VIDEO)
    print("Size:", VIDEO.stat().st_size, "bytes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with VIDEO.open("rb") as handle:
        header = handle.read(12)

        if header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise RuntimeError("Not a standard RIFF AVI file")

        movi_start, movi_end = find_movi(
            handle,
            VIDEO.stat().st_size,
        )

        frames = []
        collect_frames(handle, movi_start, movi_end, frames)

        print("Frame chunks found:", len(frames))

        selected = [
            ("first", 0),
            ("middle", len(frames) // 2),
            ("last", len(frames) - 1),
        ]

        for label, index in selected:
            image, mismatch = decode_dib_frame(
                handle,
                frames[index],
            )

            stats = image_statistics(image)
            output_path = OUT_DIR / f"P01_{label}.pgm"

            save_pgm(output_path, image)

            physical_time = index / 20000.0

            print()
            print(f"{label.upper()} FRAME")
            print("Index:", index)
            print("Physical time, s:", physical_time)
            print("Chunk:", frames[index]["chunk_id"])
            print("Stored bytes:", frames[index]["size"])
            print("Minimum:", stats["min"])
            print("Maximum:", stats["max"])
            print("Mean:", round(stats["mean"], 3))
            print(
                "Unequal RGB channel fraction:",
                round(mismatch, 8),
            )
            print("Saved:", output_path)

    print()
    print("Проверка завершена. Терминал остаётся открытым.")


if __name__ == "__main__":
    main()
