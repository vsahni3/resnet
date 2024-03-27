from struct import unpack


def parse_fg_file(file_path):
    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Read and verify the header
        header = file.read(8).decode("utf-8")
        if header != "FRFG0001":
            raise ValueError("File format not recognized.")

        (
            geometry_basis_version,
            texture_basis_version,
            ss,
            sa,
            ts,
            ta,
            _,
            detail_texture_flag,
        ) = unpack("<8I", file.read(32))

        symmetric_shape_modes = unpack(f"<{ss}h", file.read(ss * 2))
        asymmetric_shape_modes = unpack(f"<{sa}h", file.read(sa * 2))
        symmetric_texture_modes = unpack(f"<{ts}h", file.read(ts * 2))
        asymmetric_texture_modes = (
            unpack(f"<{ta}h", file.read(ta * 2)) if ta > 0 else tuple()
        )

        detail_texture_data = None
        if detail_texture_flag == 1:
            detail_texture_size = unpack("<I", file.read(4))[0]
            detail_texture_data = file.read(detail_texture_size)

        parsed_data = {
            "header": header,
            "geometry_basis_version": geometry_basis_version,
            "texture_basis_version": texture_basis_version,
            "symmetric_shape_modes": symmetric_shape_modes,
            "asymmetric_shape_modes": asymmetric_shape_modes,
            "symmetric_texture_modes": symmetric_texture_modes,
            "asymmetric_texture_modes": asymmetric_texture_modes,
            "detail_texture_flag": detail_texture_flag,
            "detail_texture_data": detail_texture_data,
            "ss": ss,
            "sa": sa,
            "ts": ts,
            "ta": ta,
        }

        return parsed_data