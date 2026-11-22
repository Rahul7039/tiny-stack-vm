// A tiny stack virtual machine for make.fun's tiny stack ISA, version 1.
// For every instance <id> in the input directory it runs <id>/program.bin on <id>/input.bin and
// writes <id>/stdout.bin (exactly the bytes the program wrote) and <id>/exit.txt (its exit code:
// 0 on HALT, the EXIT operand's low byte, or a trap's code). Values are 32-bit integers wrapping
// modulo 2^32; the data stack holds at most 256 values, the call stack 64 return addresses,
// memory 256 zeroed words; input is read byte by byte, -1 past its end; at most 1,000,000
// instructions and 65,536 output bytes. A trap ends the program at once with its code, keeping
// the output already written. Traps: 2 stack underflow, 3 stack overflow, 4 division by zero,
// 5 unknown opcode, 6 memory range, 7 call stack overflow, 8 empty RET, 9 pc outside the
// program or a cut-off immediate, 10 step limit, 11 output limit. Per step the checks fire in
// that order.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Opcode table: byte, values popped, the stack's net change; rows in the ISA's order, HALT to
// EXIT. A byte with no row traps 5. Immediate lengths: PUSH8 1, PUSH32 4, jumps and CALL 2.
const TABLE = [
  0x00, 0, 0, 0x01, 0, 1, 0x02, 0, 1, 0x03, 1, -1, 0x04, 1, 1, 0x05, 2, 0, 0x06, 2, 1,
  0x07, 3, 0, 0x10, 2, -1, 0x11, 2, -1, 0x12, 2, -1, 0x13, 2, -1, 0x14, 2, -1, 0x15, 2, -1,
  0x16, 2, -1, 0x17, 1, 0, 0x18, 2, -1, 0x19, 2, -1, 0x1a, 2, -1, 0x1b, 1, 0, 0x1c, 2, -1,
  0x1d, 2, -1, 0x1e, 2, -1, 0x20, 2, -1, 0x21, 2, -1, 0x22, 2, -1, 0x30, 0, 0, 0x31, 1, -1,
  0x32, 1, -1, 0x33, 0, 0, 0x34, 0, 0, 0x40, 1, 0, 0x41, 2, -2, 0x50, 1, -1, 0x51, 1, -1,
  0x52, 0, 1, 0x53, 1, -1,
];
const IMM = new Int8Array(256).fill(-1);
const POPS = new Int8Array(256);
const DELTA = new Int8Array(256);
for (let i = 0; i < TABLE.length; i += 3) {
  const byte = TABLE[i];
  IMM[byte] = byte === 1 ? 1 : byte === 2 ? 4 : byte >= 0x30 && byte <= 0x33 ? 2 : 0;
  POPS[byte] = TABLE[i + 1];
  DELTA[byte] = TABLE[i + 2];
}

// The binary operations, by opcode (a under, b on top); the caller traps division by zero first.
const BIN = {
  0x10: (a, b) => (a + b) | 0,
  0x11: (a, b) => (a - b) | 0,
  0x12: (a, b) => Math.imul(a, b), // MUL: the exact product's low 32 bits
  0x13: (a, b) => (a === -2147483648 && b === -1 ? a : (a / b) | 0), // DIV: truncates toward zero
  0x14: (a, b) => (b === -1 ? 0 : (a % b) | 0), // MOD: the sign of a; x % -1 is 0
  0x15: (a, b) => Math.floor((a >>> 0) / (b >>> 0)) | 0,
  0x16: (a, b) => ((a >>> 0) % (b >>> 0)) | 0,
  0x18: (a, b) => a & b,
  0x19: (a, b) => a | b,
  0x1a: (a, b) => a ^ b,
  0x1c: (a, b) => a << (b & 31), // SHL: only the count's low 5 bits
  0x1d: (a, b) => a >> (b & 31), // SHR: arithmetic, the sign bit copied in
  0x1e: (a, b) => (a >>> (b & 31)) | 0, // SHRU: logical, read back as signed
  0x20: (a, b) => (a === b ? 1 : 0),
  0x21: (a, b) => (a < b ? 1 : 0),
  0x22: (a, b) => ((a >>> 0) < (b >>> 0) ? 1 : 0), // LTU
};

// Machine state, allocated once; per program only the memory is re-zeroed.
const D = new Int32Array(256);
const C = new Int32Array(64);
const M = new Int32Array(256);
const OUT = new Uint8Array(65_536);
let ol = 0;

// Runs one program; returns its exit code. Its output is OUT[0..ol).
function run(P, INP) {
  const n = P.length;
  let sp = 0, csp = 0, pc = 0, steps = 0, ip = 0;
  M.fill(0);
  ol = 0;
  for (;;) {
    if (steps === 1_000_000) return 10;
    if (pc < 0 || pc >= n) return 9;
    const op = P[pc];
    const imm = IMM[op];
    if (imm < 0) return 5;
    if (pc + imm >= n) return 9;
    steps++;
    const next = pc + 1 + imm;
    if (sp < POPS[op]) return 2;
    if (sp + DELTA[op] > 256) return 3;
    pc = next;
    switch (op) {
    case 0x00:
      return 0;
    case 0x01: // PUSH8: the byte sign-extended from 8 bits
      D[sp++] = (P[next - 1] << 24) >> 24;
      break;
    case 0x02: // PUSH32: four bytes, little-endian
      D[sp++] = P[next - 4] | (P[next - 3] << 8) | (P[next - 2] << 16) | (P[next - 1] << 24);
      break;
    case 0x03: // POP
      sp--;
      break;
    case 0x04: // DUP
      D[sp] = D[sp - 1];
      sp++;
      break;
    case 0x05: { // SWAP
      const t = D[sp - 1];
      D[sp - 1] = D[sp - 2];
      D[sp - 2] = t;
      break;
    }
    case 0x06: // OVER
      D[sp] = D[sp - 2];
      sp++;
      break;
    case 0x07: { // ROT: ( a b c -- b c a )
      const a = D[sp - 3];
      D[sp - 3] = D[sp - 2];
      D[sp - 2] = D[sp - 1];
      D[sp - 1] = a;
      break;
    }
    case 0x17: // NEG: wraps, so -(-2147483648) is -2147483648
      D[sp - 1] = -D[sp - 1] | 0;
      break;
    case 0x1b:
      D[sp - 1] = ~D[sp - 1];
      break;
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
    case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
    case 0x20: case 0x21: case 0x22: { // binary ops: ( a b -- r ), b on top
      const b = D[sp - 1];
      if (op >= 0x13 && op <= 0x16 && b === 0) return 4;
      D[sp - 2] = BIN[op](D[sp - 2], b);
      sp--;
      break;
    }
    case 0x30: case 0x31: case 0x32: { // JMP JZ JNZ: target = next address + signed offset
      const rel = ((P[next - 2] | (P[next - 1] << 8)) << 16) >> 16;
      let take = op === 0x30;
      if (!take) {
        const v = D[--sp];
        take = op === 0x31 ? v === 0 : v !== 0;
      }
      if (take) pc = next + rel;
      break;
    }
    case 0x33: // CALL: an unsigned absolute address
      if (csp === 64) return 7;
      C[csp++] = next;
      pc = P[next - 2] | (P[next - 1] << 8);
      break;
    case 0x34:
      if (csp === 0) return 8;
      pc = C[--csp];
      break;
    case 0x40: { // LOAD
      const addr = D[sp - 1];
      if (addr < 0 || addr > 255) return 6;
      D[sp - 1] = M[addr];
      break;
    }
    case 0x41: { // STORE: ( v addr -- )
      const addr = D[sp - 1];
      if (addr < 0 || addr > 255) return 6;
      M[addr] = D[sp - 2];
      sp -= 2;
      break;
    }
    case 0x50: { // PRINT: decimal ASCII, written whole or not at all
      const s = String(D[sp - 1]);
      if (ol + s.length > 65_536) return 11;
      sp--;
      for (let i = 0; i < s.length; i++) OUT[ol++] = s.charCodeAt(i);
      break;
    }
    case 0x51: // PUTC: the low byte
      if (ol + 1 > 65_536) return 11;
      OUT[ol++] = D[--sp] & 255;
      break;
    case 0x52: // IN: the next byte, or -1 past the end
      D[sp++] = ip < INP.length ? INP[ip++] : -1;
      break;
    case 0x53: // EXIT: the low byte
      return D[--sp] & 255;
    }
  }
}

const read = (path) => {
  try {
    return readFileSync(path);
  } catch {
    return new Uint8Array(0);
  }
};
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const code = run(read(join(I, id, 'program.bin')), read(join(I, id, 'input.bin')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), OUT.subarray(0, ol));
  writeFileSync(join(O, id, 'exit.txt'), `${code}\n`);
}
