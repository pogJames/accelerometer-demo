"""Minimal Modbus RTU client — FC04 (read input registers) and FC06
(write single register) only, directly over pyserial.

Replaces pymodbus on the hot read path. pymodbus is general-purpose
(framers, codecs, transports, retries, server support); for our use
case — single slave, fixed device_id=1, only FC04/FC06, no error
recovery beyond reconnect — that abstraction stack costs ~1 ms of
Python per call. This module is ~100 lines, all of it the actual wire
work; it shaves the Python overhead and gives us full insight when
something goes wrong.
"""

import struct


# Precomputed Modbus RTU CRC-16 table (poly 0xA001 = reversed 0x8005).
# Built once at import so the per-byte CRC loop is a single table lookup
# instead of an 8-iteration bit-shift in Python.
_CRC_TABLE = []
for _i in range(256):
    _crc = _i
    for _ in range(8):
        _crc = (_crc >> 1) ^ 0xA001 if _crc & 1 else _crc >> 1
    _CRC_TABLE.append(_crc)
_CRC_TABLE = tuple(_CRC_TABLE)


class ModbusError(Exception):
    """Raised on timeout, CRC mismatch, exception response, or framing error."""


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


def read_input_registers(ser, slave_id: int, address: int, count: int):
    """FC04 — read `count` input registers starting at `address`.
    Returns a list of int (each register decoded as big-endian uint16).
    Raises ModbusError on any wire-level problem."""
    # Request: slave + func + addr_hi + addr_lo + count_hi + count_lo + crc_lo + crc_hi
    req = struct.pack('>BBHH', slave_id, 0x04, address, count)
    req += struct.pack('<H', _crc16(req))   # Modbus CRC is little-endian on wire

    # Discard anything left over from a previous (possibly failed) txn so
    # we don't sync onto stale bytes. pyserial does this at the OS level.
    ser.reset_input_buffer()
    ser.write(req)

    # Expected response: slave + func + byte_count + data(2*count) + crc(2)
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
        # Exception response: high bit set on func code; data byte is exception code.
        raise ModbusError(f"modbus exception code 0x{resp[2]:02x}")
    if resp[1] != 0x04:
        raise ModbusError(f"wrong func: got 0x{resp[1]:02x}, expected 0x04")
    if resp[2] != 2 * count:
        raise ModbusError(f"wrong byte count: got {resp[2]}, expected {2 * count}")

    # CRC over everything except the trailing 2 CRC bytes.
    expected_crc = _crc16(resp[:-2])
    actual_crc = resp[-1] << 8 | resp[-2]   # little-endian on wire
    if expected_crc != actual_crc:
        raise ModbusError(
            f"CRC mismatch: got 0x{actual_crc:04x}, expected 0x{expected_crc:04x}"
        )

    # Decode N big-endian uint16s in one struct.unpack call (faster than a
    # Python-level loop, and avoids creating a numpy array we'd just discard
    # — sensor_reader does its own typed conversion straight from this list).
    return list(struct.unpack(f'>{count}H', resp[3:3 + 2 * count]))


def write_single_register(ser, slave_id: int, address: int, value: int):
    """FC06 — write `value` (16-bit) to register `address`.
    Response is a byte-for-byte echo of the request; we validate that
    the echo matches what we sent and raise on any mismatch."""
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
