import struct


# Modbus RTU CRC-16 table (poly 0xA001), built once so per-byte CRC is a
# table lookup instead of an 8-iteration bit-shift in Python.
_CRC_TABLE = []
for _i in range(256):
    _crc = _i
    for _ in range(8):
        _crc = (_crc >> 1) ^ 0xA001 if _crc & 1 else _crc >> 1
    _CRC_TABLE.append(_crc)
_CRC_TABLE = tuple(_CRC_TABLE)


class ModbusError(Exception):
    pass


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


# FC04 — read `count` input registers at `address`; returns list of int.
def read_input_registers(ser, slave_id: int, address: int, count: int):
    req = struct.pack('>BBHH', slave_id, 0x04, address, count)
    req += struct.pack('<H', _crc16(req))   # CRC little-endian on wire

    ser.reset_input_buffer()
    ser.write(req)

    expected_len = 5 + 2 * count
    resp = ser.read(expected_len)
    if len(resp) != expected_len:
        raise ModbusError(
            f"short read: got {len(resp)} of {expected_len} bytes "
            f"(slave={slave_id}, addr=0x{address:04x}, count={count})"
        )

    if resp[0] != slave_id:
        raise ModbusError(f"wrong slave_id: got {resp[0]}, expected {slave_id}")
    if resp[1] & 0x80:      # high bit set → exception response, data byte is code
        raise ModbusError(f"modbus exception code 0x{resp[2]:02x}")
    if resp[1] != 0x04:
        raise ModbusError(f"wrong func: got 0x{resp[1]:02x}, expected 0x04")
    if resp[2] != 2 * count:
        raise ModbusError(f"wrong byte count: got {resp[2]}, expected {2 * count}")

    expected_crc = _crc16(resp[:-2])
    actual_crc = resp[-1] << 8 | resp[-2]   # little-endian on wire
    if expected_crc != actual_crc:
        raise ModbusError(
            f"CRC mismatch: got 0x{actual_crc:04x}, expected 0x{expected_crc:04x}"
        )

    return list(struct.unpack(f'>{count}H', resp[3:3 + 2 * count]))


# FC03 — read `count` holding registers at `address`; returns list of int.
def read_holding_registers(ser, slave_id: int, address: int, count: int):
    req = struct.pack('>BBHH', slave_id, 0x03, address, count)
    req += struct.pack('<H', _crc16(req))

    ser.reset_input_buffer()
    ser.write(req)

    expected_len = 5 + 2 * count
    resp = ser.read(expected_len)
    if len(resp) != expected_len:
        raise ModbusError(
            f"short read: got {len(resp)} of {expected_len} bytes "
            f"(slave={slave_id}, addr=0x{address:04x}, count={count})"
        )

    if resp[0] != slave_id:
        raise ModbusError(f"wrong slave_id: got {resp[0]}, expected {slave_id}")
    if resp[1] & 0x80:
        raise ModbusError(f"modbus exception code 0x{resp[2]:02x}")
    if resp[1] != 0x03:
        raise ModbusError(f"wrong func: got 0x{resp[1]:02x}, expected 0x03")
    if resp[2] != 2 * count:
        raise ModbusError(f"wrong byte count: got {resp[2]}, expected {2 * count}")

    expected_crc = _crc16(resp[:-2])
    actual_crc = resp[-1] << 8 | resp[-2]
    if expected_crc != actual_crc:
        raise ModbusError(
            f"CRC mismatch: got 0x{actual_crc:04x}, expected 0x{expected_crc:04x}"
        )

    return list(struct.unpack(f'>{count}H', resp[3:3 + 2 * count]))


# FC06 — write 16-bit `value` to `address`; response echoes the request.
def write_single_register(ser, slave_id: int, address: int, value: int):
    req = struct.pack('>BBHH', slave_id, 0x06, address, value & 0xFFFF)
    req += struct.pack('<H', _crc16(req))

    ser.reset_input_buffer()
    ser.write(req)

    resp = ser.read(8)
    if len(resp) != 8:
        raise ModbusError(f"short write response: got {len(resp)} of 8 bytes")
    if resp[1] & 0x80:
        raise ModbusError(f"modbus exception code 0x{resp[2]:02x}")
    if resp != req:
        raise ModbusError(f"echo mismatch: sent {req.hex()}, got {resp.hex()}")
