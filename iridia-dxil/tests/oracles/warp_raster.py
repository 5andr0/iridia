"""Record which pixels Direct3D 12 draws for points and aliased lines on WARP and on the machine's GPU, for
tests/exec_warp_draw.rs's point and line tests (which need neither).

usage: python -I warp_raster.py <warp.exe> <tests/fixtures> <scratch folder> <adapter name>

warp.exe is the WARP oracle of primase-oracles fa44129 or later (the d3d12draw form's `topology` and `adapter` lines);
<adapter name> is what the `adapter hardware` run draws on (the machine's first adapter; the oracle does not report
it), written into that file's header. Every case draws one primitive through tests/fixtures/draw_vs.dxil and
draw_colour.dxil into a 16 x 16 target, at positions on a grid of 1/8 to 1/256 of a pixel, so the snapped positions
are the drawn ones exactly:

- 600 random points, inside and outside the clip volume;
- 3,000 random lines, ends on a 1/8 and a 1/256 grid, 30% of them at 45 degrees or along an axis;
- 2,400 lines at 45 degrees, in the four directions in turn, ends on a 1/16 grid: many lie along diamond edges;
- 4,640 lines ending at every 1/16-pixel point of pixel (8, 8)'s diamond, in each of the eight octants, at slopes
  64/1024, 512/1024, 960/1024 and 1;
- 600 lines on one 45-degree line along diamond edges, the start slid along it by 1/128 of a pixel at a time.

It writes tests/fixtures/warp_raster_warp.txt and warp_raster_hardware.txt, the same cases in the same order:

    # adapter <warp | hardware: name>
    point <x> <y> <z> <pixels>
    line <x0> <y0> <x1> <y1> <pixels>

with coordinates in 1/256 of a pixel (z too), and the pixels drawn as two hex digits, column then row. The hardware
run must draw differently from WARP somewhere, or the script stops: a run the oracle sent to WARP cannot pass for a GPU.
"""
import json
import os
import random
import struct
import subprocess
import sys

S = 16


def hx(x):
    return '%08x' % struct.unpack('<I', struct.pack('<f', x))[0]


def clip_of(x, y, z=0.5):
    return [x / S * 2 - 1, 1 - y / S * 2, z, 1.0]


def lines(seed, count, grid):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        pick = lambda: rng.randrange(0, S * grid + 1) / grid
        a, b = (pick(), pick()), (pick(), pick())
        if rng.random() < 0.3:  # 45 degree and axis-aligned lines
            s = rng.choice([-1, 1]) * rng.randrange(1, 8 * grid) / grid
            b = (a[0] + s, a[1] + rng.choice([-1, 0, 1]) * s) if rng.random() < 0.7 else (a[0], a[1] + s)
        out.append((a, b))
    return out


def diagonals(seed, count):
    rng = random.Random(seed)
    out = []
    for k in range(count):
        sx, sy = [(1, 1), (-1, 1), (1, -1), (-1, -1)][k % 4]
        a = (rng.randrange(32, 224) / 16, rng.randrange(32, 224) / 16)
        n = rng.randrange(1, 96) / 16
        out.append((a, (a[0] + sx * n, a[1] + sy * n)))
    return out


def ends():
    """Lines 4 pixels long along the major axis ending inside pixel (8, 8)'s diamond."""
    centre = 8.5
    out = []
    for major in (0, 1):
        for sM in (1, -1):
            for sm in (1, -1):
                for n in (64, 512, 960, 1024):
                    for em in range(-128, 129, 16):
                        for en in range(-128, 129, 16):
                            if abs(em) + abs(en) > 128:
                                continue
                            e = [centre + em / 256, centre + en / 256]
                            d = [0.0, 0.0]
                            d[major] = sM * 4
                            d[1 - major] = sm * 4 * n / 1024
                            out.append(((e[0] - d[0], e[1] - d[1]), tuple(e)))
    return out


def slide():
    """One 45-degree line along diamond edges (x + y = 20.5 pixels), its start slid along it."""
    return [((3360 / 256 + k / 256, 1888 / 256 - k / 256), (1888 / 256, 3360 / 256)) for k in range(-900, 300, 2)]


def points(seed, count):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        grid = rng.choice([2, 8, 256])
        x, y = (rng.randrange(-grid, (S + 1) * grid + 1) / grid for _ in range(2))
        z = rng.choice([0.5, 0.0, 1.0, -1 / 256, 1 + 1 / 256])
        out.append((x, y, z))
    return out


def case(name, fixtures, topology, adapter, vertices):
    t = f'case {name}\napi d3d12draw\nvs {os.path.join(fixtures, "draw_vs.dxil")}\n'
    t += f'ps {os.path.join(fixtures, "draw_colour.dxil")}\nsize {S} {S}\ncull none\nfront 1\ntopology {topology}\n'
    t += f'adapter {adapter}\n'
    t += 'layout POSITION 0 f32x4 0\nlayout COLOR 0 f32x4 16\nlayout TEXCOORD 0 f32x2 32\nlayout TEXCOORD 1 f32x4 40\n'
    t += 'stride 56\n'
    for pos in vertices:
        t += 'vertex ' + ' '.join(hx(v) for v in pos + [1, 0, 0, 1, 0, 0, 1, 1.5, 0, 1]) + '\n'
    t += 'index ' + ' '.join('%x' % i for i in range(len(vertices))) + '\n'
    t += 'vcb 0 2\nrow ' + ' '.join([hx(1.0)] * 4) + '\nrow ' + ' '.join([hx(0.0)] * 4) + '\n'
    return t + 'target 0 float\nclear 0 ' + ' '.join([hx(0.0)] * 4) + '\nend\n'


def main():
    warp, fixtures, work, name = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    prims = [('point', 'pointlist', [p]) for p in points(3, 600)]
    every = lines(1, 1500, 8) + lines(2, 1500, 256) + diagonals(7, 2400) + ends() + slide()
    prims += [('line', 'linelist', [a, b]) for a, b in every]
    drawn = {}
    for adapter in ('warp', 'hardware'):
        text = ''.join(
            case(f'{kind[0]}{k}', fixtures, topology, adapter, [clip_of(*v) for v in vs])
            for k, (kind, topology, vs) in enumerate(prims)
        )
        inp, out = os.path.join(work, f'{adapter}.txt'), os.path.join(work, f'{adapter}.jsonl')
        with open(inp, 'w', newline='\n') as f:
            f.write(text)
        subprocess.run([warp, inp, out], check=True)
        results = [json.loads(l) for l in open(out)]
        assert len(results) == len(prims)
        drawn[adapter] = []
        for r in results:
            assert 'error' not in r, r
            w = [int(x, 16) for x in r['targets'][0]]
            drawn[adapter].append(sorted((p % S, p // S) for p in range(S * S) if any(w[p * 4:p * 4 + 4])))
    differ = sum(a != b for a, b in zip(drawn['warp'], drawn['hardware']))
    assert differ, 'the hardware run drew what WARP drew on every case: it ran on WARP'
    print(f'{len(prims)} primitives; the hardware draws differently from WARP on {differ}')
    for adapter, header in (('warp', 'warp'), ('hardware', f'hardware: {name}')):
        text = f'# adapter {header}\n'
        for (kind, _, vs), pixels in zip(prims, drawn[adapter]):
            coords = [round(c * 256) for v in vs for c in v]
            text += f'{kind} ' + ' '.join(map(str, coords)) + ''.join(' %x%x' % p for p in pixels) + '\n'
        with open(os.path.join(fixtures, f'warp_raster_{adapter}.txt'), 'w', newline='\n') as f:
            f.write(text)


main()
