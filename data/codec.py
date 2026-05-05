import struct
import reedsolo
from typing import Tuple


LENGTH_PREFIX_BYTES = 4
FILENAME_LENGTH_BYTES = 2
MAX_FILENAME_LENGTH = 65535
RS_BLOCK_DATA = 223
RS_BLOCK_PARITY = 32
RS_BLOCK_TOTAL = 255
ENCODED_SIZE_PREFIX_BYTES = 4
PAYLOAD_SIZE_PREFIX_BYTES = 4


def _rs_encode(payload_bytes: bytes) -> bytes:
    """使用Reed-Solomon纠错码对字节序列进行分块编码

    将输入字节按RS_BLOCK_DATA(223)字节分块，每块补零到223字节后RS编码为完整255字节。
    最后一块不足223字节时也补零编码，保持所有块统一为255字节。

    Args:
        payload_bytes: 待编码的原始字节序列

    Returns:
        RS编码后的字节序列（长度为255的整数倍）
    """
    rs = reedsolo.RSCodec(RS_BLOCK_PARITY)
    encoded_blocks = []
    offset = 0
    while offset < len(payload_bytes):
        chunk = payload_bytes[offset:offset + RS_BLOCK_DATA]
        if len(chunk) < RS_BLOCK_DATA:
            chunk = chunk + b"\x00" * (RS_BLOCK_DATA - len(chunk))
        encoded_blocks.append(rs.encode(chunk))
        offset += RS_BLOCK_DATA
    return b"".join(encoded_blocks)


def _rs_decode(encoded_bytes: bytes, data_byte_count: int) -> bytes:
    """使用Reed-Solomon纠错码对字节序列进行分块解码

    每块255字节解码为223字节，自动纠正每块中最多16字节的错误。

    Args:
        encoded_bytes: RS编码后的字节序列（长度应为255的整数倍）
        data_byte_count: 原始数据的总字节数，用于确定返回的字节长度

    Returns:
        解码纠错后的原始字节序列

    Raises:
        ValueError: 当RS解码失败（错误超出纠错能力）时抛出
    """
    rs = reedsolo.RSCodec(RS_BLOCK_PARITY)
    num_blocks = len(encoded_bytes) // RS_BLOCK_TOTAL
    decoded_blocks = []

    for i in range(num_blocks):
        block = encoded_bytes[i * RS_BLOCK_TOTAL:(i + 1) * RS_BLOCK_TOTAL]
        try:
            decoded = rs.decode(block)
            decoded_blocks.append(decoded[0])
        except reedsolo.ReedSolomonError as e:
            raise ValueError(f"RS解码失败: 第{i}块纠错失败: {e}")

    full_decoded = b"".join(decoded_blocks)
    return full_decoded[:data_byte_count]


def _bits_to_bytes(bits: str) -> bytes:
    """将比特流字符串转换为字节序列

    Args:
        bits: 比特流字符串

    Returns:
        对应的字节序列
    """
    byte_list = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            chunk = chunk + "0" * (8 - len(chunk))
        byte_list.append(int(chunk, 2))
    return bytes(byte_list)


def text_to_bits(text: str, fixed_length_bits: int = None) -> str:
    """将UTF-8文本编码为比特流字符串，使用Reed-Solomon纠错码保护

    编码格式: [encoded_size_4B][RS(text_bytes || length_4B)][padding_zeros]
    - encoded_size: 4字节大端序，记录RS编码后的数据字节数（用于解码时定位真实数据边界）
    - text_bytes || length_4B: 文本的UTF-8字节 + 4字节大端序文本长度（长度在末尾）
    - padding_zeros: 填充到fixed_length_bits的零字节

    Args:
        text: 待编码的文本字符串
        fixed_length_bits: 如果指定，将比特流填充/截断到固定长度

    Returns:
        比特流字符串（仅包含'0'和'1'）
    """
    text_bytes = text.encode("utf-8")
    length_prefix = struct.pack(">I", len(text_bytes))
    payload = text_bytes + length_prefix
    encoded_bytes = _rs_encode(payload)
    encoded_size_header = struct.pack(">I", len(encoded_bytes))
    payload_size_header = struct.pack(">I", len(payload))
    full_data = encoded_size_header + payload_size_header + encoded_bytes
    data_bits = "".join(format(b, "08b") for b in full_data)

    if fixed_length_bits is not None:
        if len(data_bits) > fixed_length_bits:
            data_bits = data_bits[:fixed_length_bits]
        else:
            data_bits = data_bits + "0" * (fixed_length_bits - len(data_bits))

    return data_bits


def bits_to_text(bits: str) -> str:
    """将比特流字符串解码为UTF-8文本，使用Reed-Solomon纠错码自动纠正错误

    解码流程:
    1. 比特流→字节→读取前4字节获取RS编码数据大小
    2. 提取编码数据→RS解码→从末尾提取长度→提取文本→UTF-8解码

    Args:
        bits: 比特流字符串（仅包含'0'和'1'）

    Returns:
        解码后的文本字符串

    Raises:
        ValueError: 当RS解码失败或数据格式无效时抛出
    """
    raw_bytes = _bits_to_bytes(bits)

    header_size = ENCODED_SIZE_PREFIX_BYTES + PAYLOAD_SIZE_PREFIX_BYTES
    min_size = header_size + RS_BLOCK_TOTAL
    if len(raw_bytes) < min_size:
        raise ValueError(f"比特流长度不足: {len(raw_bytes)} < {min_size}")

    encoded_size = struct.unpack(">I", raw_bytes[:ENCODED_SIZE_PREFIX_BYTES])[0]
    payload_size = struct.unpack(">I", raw_bytes[ENCODED_SIZE_PREFIX_BYTES:header_size])[0]

    if encoded_size == 0 or encoded_size > len(raw_bytes) - header_size:
        raise ValueError(f"无效的编码数据大小: {encoded_size}")
    if payload_size == 0 or payload_size > encoded_size // RS_BLOCK_TOTAL * RS_BLOCK_DATA:
        raise ValueError(f"无效的payload大小: {payload_size}")

    encoded_bytes = raw_bytes[header_size:header_size + encoded_size]
    decoded_bytes = _rs_decode(encoded_bytes, payload_size)

    text_length = struct.unpack(">I", decoded_bytes[-LENGTH_PREFIX_BYTES:])[0]

    if text_length == 0 or text_length > payload_size - LENGTH_PREFIX_BYTES:
        raise ValueError(f"无效的文本长度: {text_length}")

    text_bytes = decoded_bytes[-LENGTH_PREFIX_BYTES - text_length:-LENGTH_PREFIX_BYTES]
    return text_bytes.decode("utf-8")


def file_to_bits(file_path: str) -> Tuple[str, str]:
    """将文件读取为比特流字符串，使用Reed-Solomon纠错码保护

    编码格式: [encoded_size_4B][RS(filename_length_2B + filename_bytes + content_length_4B + content_bytes)][padding]

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
    payload = filename_length_prefix + filename_bytes + content_length_prefix + file_bytes

    encoded_bytes = _rs_encode(payload)
    encoded_size_header = struct.pack(">I", len(encoded_bytes))
    payload_size_header = struct.pack(">I", len(payload))
    full_data = encoded_size_header + payload_size_header + encoded_bytes
    bits = "".join(format(b, "08b") for b in full_data)
    return bits, filename


def bits_to_file(bits: str, output_dir: str = ".") -> str:
    """将比特流字符串解码并保存为文件，使用Reed-Solomon纠错码自动纠正错误

    解码流程:
    1. 比特流→字节→读取前4字节获取RS编码数据大小
    2. 提取编码数据→RS解码
    3. 顺序解析: filename_length → filename → content_length → content

    Args:
        bits: 比特流字符串
        output_dir: 输出目录

    Returns:
        保存的文件路径

    Raises:
        ValueError: 当RS解码失败或数据格式无效时抛出
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    raw_bytes = _bits_to_bytes(bits)

    header_size = ENCODED_SIZE_PREFIX_BYTES + PAYLOAD_SIZE_PREFIX_BYTES
    if len(raw_bytes) < header_size:
        raise ValueError(f"比特流长度不足")

    encoded_size = struct.unpack(">I", raw_bytes[:ENCODED_SIZE_PREFIX_BYTES])[0]
    payload_size = struct.unpack(">I", raw_bytes[ENCODED_SIZE_PREFIX_BYTES:header_size])[0]
    encoded_bytes = raw_bytes[header_size:header_size + encoded_size]
    decoded_bytes = _rs_decode(encoded_bytes, payload_size)

    offset = 0
    filename_length = struct.unpack(">H", decoded_bytes[offset:offset + FILENAME_LENGTH_BYTES])[0]
    offset += FILENAME_LENGTH_BYTES

    filename_bytes = decoded_bytes[offset:offset + filename_length]
    filename = filename_bytes.decode("utf-8")
    offset += filename_length

    content_length = struct.unpack(">I", decoded_bytes[offset:offset + LENGTH_PREFIX_BYTES])[0]
    offset += LENGTH_PREFIX_BYTES

    file_bytes = decoded_bytes[offset:offset + content_length]

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

    考虑Reed-Solomon校验开销和编码数据大小头。
    每223字节数据需要223+32=255字节编码空间（完整块），
    不足223字节的最后一块需要data+32字节编码空间。
    额外开销: 4字节编码大小头 + 4字节文本长度。

    Args:
        data_depth: 数据深度（每像素比特数）
        image_size: 图像尺寸

    Returns:
        可嵌入的最大文本字节数
    """
    total_bits = data_depth * image_size * image_size
    total_bytes = total_bits // 8
    header_size = ENCODED_SIZE_PREFIX_BYTES + PAYLOAD_SIZE_PREFIX_BYTES
    available = total_bytes - header_size
    num_full_blocks = available // RS_BLOCK_TOTAL
    remaining = available % RS_BLOCK_TOTAL
    data_bytes = num_full_blocks * RS_BLOCK_DATA + max(remaining - RS_BLOCK_PARITY, 0)
    return max(data_bytes - LENGTH_PREFIX_BYTES, 0)


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
