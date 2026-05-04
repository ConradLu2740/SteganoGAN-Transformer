import struct
from typing import Tuple


LENGTH_PREFIX_BYTES = 4
FILENAME_LENGTH_BYTES = 2
MAX_FILENAME_LENGTH = 65535


def text_to_bits(text: str, fixed_length_bits: int = None) -> str:
    """将UTF-8文本编码为比特流字符串，使用三重冗余长度前缀提高鲁棒性

    将长度前缀重复3次放在开头，提高解码成功率。

    Args:
        text: 待编码的文本字符串
        fixed_length_bits: 如果指定，将比特流填充/截断到固定长度

    Returns:
        比特流字符串（仅包含'0'和'1'）
    """
    text_bytes = text.encode("utf-8")
    length_prefix = struct.pack(">I", len(text_bytes))
    length_bits = "".join(format(b, "08b") for b in length_prefix)
    text_bits = "".join(format(b, "08b") for b in text_bytes)

    # 格式: [长度前缀×3][文本比特] - 三重冗余
    data_bits = length_bits * 3 + text_bits

    if fixed_length_bits is not None:
        if len(data_bits) > fixed_length_bits:
            # 保留长度前缀，截断文本
            data_bits = data_bits[:fixed_length_bits]
        else:
            data_bits = data_bits + "0" * (fixed_length_bits - len(data_bits))

    return data_bits


def bits_to_text(bits: str) -> str:
    """将比特流字符串解码为UTF-8文本，使用三重冗余长度前缀

    Args:
        bits: 比特流字符串（仅包含'0'和'1'）

    Returns:
        解码后的文本字符串
    """
    length_bits_count = LENGTH_PREFIX_BYTES * 8
    redundancy_bits = length_bits_count * 3

    if len(bits) < redundancy_bits:
        raise ValueError(f"比特流长度不足: {len(bits)} < {redundancy_bits}")

    # 读取三个长度前缀，投票决定
    len1 = struct.unpack(">I", int(bits[:length_bits_count], 2).to_bytes(LENGTH_PREFIX_BYTES, "big"))[0]
    len2 = struct.unpack(">I", int(bits[length_bits_count:length_bits_count*2], 2).to_bytes(LENGTH_PREFIX_BYTES, "big"))[0]
    len3 = struct.unpack(">I", int(bits[length_bits_count*2:length_bits_count*3], 2).to_bytes(LENGTH_PREFIX_BYTES, "big"))[0]

    # 投票：选择出现次数最多的长度
    lengths = [len1, len2, len3]
    text_length = max(set(lengths), key=lengths.count)

    # 如果投票结果不合理，尝试单个值
    if text_length == 0 or text_length > 10000:
        for l in lengths:
            if 0 < l < 1000:
                text_length = l
                break

    if text_length == 0 or text_length > 10000:
        # 兜底：直接解码可见内容
        try:
            all_bytes = bytes(int(bits[i:i + 8], 2) for i in range(redundancy_bits, len(bits), 8))
            filtered = "".join(c for c in all_bytes.decode("utf-8", errors="replace") if c.isprintable() or c in " \t\n")
            return filtered[:100]
        except Exception:
            pass
        return ""

    total_bits = redundancy_bits + text_length * 8

    if len(bits) < total_bits:
        try:
            text_bytes = bytes(int(bits[i:i + 8], 2) for i in range(redundancy_bits, len(bits), 8))
            return text_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ValueError(f"比特流长度不足: {len(bits)} < {total_bits}")

    text_bytes = bytes(int(bits[i:i + 8], 2) for i in range(redundancy_bits, total_bits, 8))
    return text_bytes.decode("utf-8", errors="replace")


def file_to_bits(file_path: str) -> Tuple[str, str]:
    """将文件读取为比特流字符串，前缀为文件名长度和文件内容长度

    Args:
        file_path: 文件路径

    Returns:
        (比特流字符串, 文件名字符串)
    """
    import os
    filename = os.path.basename(file_path)
    filename_bytes = filename.encode("utf-8")

    if len(filename_bytes) > MAX_FILENAME_LENGTH:
        raise ValueError(f"文件名过长: {len(filename_bytes)} > {MAX_FILENAME_LENGTH}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    filename_length_prefix = struct.pack(">H", len(filename_bytes))
    content_length_prefix = struct.pack(">I", len(file_bytes))
    full_bytes = filename_length_prefix + filename_bytes + content_length_prefix + file_bytes

    bits = "".join(format(b, "08b") for b in full_bytes)
    return bits, filename


def bits_to_file(bits: str, output_dir: str = ".") -> str:
    """将比特流字符串解码并保存为文件

    Args:
        bits: 比特流字符串
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    offset = 0

    filename_length = struct.unpack(">H", int(bits[offset:offset + FILENAME_LENGTH_BYTES * 8], 2).to_bytes(FILENAME_LENGTH_BYTES, "big"))[0]
    offset += FILENAME_LENGTH_BYTES * 8

    filename_bytes = bytes(int(bits[i:i + 8], 2) for i in range(offset, offset + filename_length * 8, 8))
    filename = filename_bytes.decode("utf-8")
    offset += filename_length * 8

    content_length = struct.unpack(">I", int(bits[offset:offset + LENGTH_PREFIX_BYTES * 8], 2).to_bytes(LENGTH_PREFIX_BYTES, "big"))[0]
    offset += LENGTH_PREFIX_BYTES * 8

    file_bytes = bytes(int(bits[i:i + 8], 2) for i in range(offset, offset + content_length * 8, 8))

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(file_bytes)

    return output_path


def bits_to_tensor(bits: str, shape: tuple) -> "torch.Tensor":
    """将比特流字符串转换为指定形状的PyTorch张量

    Args:
        bits: 比特流字符串
        shape: 目标张量形状 (C, H, W)

    Returns:
        值为0或1的FloatTensor
    """
    import torch
    total_bits = 1
    for s in shape:
        total_bits *= s
    bit_list = [float(b) for b in bits[:total_bits]]
    while len(bit_list) < total_bits:
        bit_list.append(0.0)
    tensor = torch.tensor(bit_list, dtype=torch.float32).reshape(shape)
    return tensor


def tensor_to_bits(tensor: "torch.Tensor") -> str:
    """将PyTorch张量转换为比特流字符串

    Args:
        tensor: 值接近0或1的张量

    Returns:
        比特流字符串
    """
    binary = (tensor > 0.5).long().flatten().tolist()
    return "".join(str(b) for b in binary)


def max_text_capacity(data_depth: int, image_size: int) -> int:
    """计算给定配置下可嵌入的最大UTF-8文本字节数

    Args:
        data_depth: 数据深度（每像素比特数）
        image_size: 图像尺寸

    Returns:
        可嵌入的最大文本字节数
    """
    total_bits = data_depth * image_size * image_size
    length_prefix_bits = LENGTH_PREFIX_BYTES * 8
    available_bits = total_bits - length_prefix_bits
    return max(available_bits // 8, 0)


def validate_text_length(text: str, data_depth: int, image_size: int) -> None:
    """校验文本长度是否超出隐写容量

    Args:
        text: 待嵌入的文本
        data_depth: 数据深度
        image_size: 图像尺寸

    Raises:
        ValueError: 文本过长时抛出异常
    """
    text_bytes = text.encode("utf-8")
    capacity = max_text_capacity(data_depth, image_size)
    if len(text_bytes) > capacity:
        raise ValueError(
            f"文本过长（{len(text_bytes)} 字节），超出隐写容量（{capacity} 字节）。"
            f"请缩短文本或增大数据深度（当前 data_depth={data_depth}, image_size={image_size}）"
        )
