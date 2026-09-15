/* Doom2 plus one chasing "thing": an 8 x 16 transparent imp billboard at Q4.4.
   The tick projects it into camera space, the sprite-column kernel compares its depth with
   the wall z-buffer, and the pixel kernel overlays nonzero sprite texels.  Screen 160 x 100,
   horizon row 50; heading is the left ray and heading + 10 is the camera centre. */
static u16 grid[256] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static u16 cost[64] = {16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534, 0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16};
static u16 sint[64] = {0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534};
static u16 htab[6] = {4, 40, 20, 13, 10, 4};
static u16 shade[5] = {7, 3, 4, 4, 5};
static u16 colmap[160] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19};
static u16 hbuf[160];
static u16 dbuf[160];
static u16 ubuf[160];
static u16 sbuf[160];
static u16 stex[160];
static u16 texture[128] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 3, 3, 3, 3, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14, 14, 14, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14, 14, 14, 14, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7};   /* 8 x 8 bricks, a bright and a dark bank */
/* The chasing imp can grow to 120 pixels high, so its texture-row reciprocal needs that range. */
static u16 recip2[121] = {1024, 1024, 512, 341, 256, 205, 171, 146, 128, 114, 102, 93, 85, 79, 73, 68, 64, 60, 57, 54, 51, 49, 47, 45, 43, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 28, 27, 26, 26, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 20, 19, 19, 19, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
static u16 sprite[128] = {
    0, 0, 15, 0, 0, 15, 0, 0,
    0, 0, 15, 15, 15, 15, 0, 0,
    0, 15, 10, 15, 15, 10, 15, 0,
    0, 15, 15, 15, 15, 15, 15, 0,
    0, 15, 9, 15, 15, 9, 15, 0,
    0, 15, 15, 10, 10, 15, 15, 0,
    0, 0, 15, 15, 15, 15, 0, 0,
    0, 15, 15, 15, 15, 15, 15, 0,
    15, 15, 15, 10, 10, 15, 15, 15,
    15, 0, 15, 10, 10, 15, 0, 15,
    0, 0, 15, 10, 10, 15, 0, 0,
    0, 0, 15, 15, 15, 15, 0, 0,
    0, 0, 15, 0, 0, 15, 0, 0,
    0, 15, 15, 0, 0, 15, 15, 0,
    0, 15, 0, 0, 0, 0, 15, 0,
    15, 15, 0, 0, 0, 0, 15, 15
};
static u16 px, py, tx, ty, heading, in, turn, fwd, f, col, s, ang, dx, dy, x, y, cell, found, dist, h, nx, ny;
static u16 tdx, tdy, xneg, yneg, bigger, tcell, pcell;
static u16 t, prow, pcol, top, bot, a, b, c, d, p, u, v, dark, hx, hy;
static u16 sdepth, side, scol, hw, sh, recip2w, sbh, su, near, covers, twohw, left;
static u16 camang, relx, rely, cc, ss, depth, sraw, sign, off, stop, sbot, sv, spix;

int main(void) {
    px = 120; py = 120; tx = 168; ty = 88; heading = 48;
    /* Projection of the initial thing (168,88): depth 3, centred column 81, half-width 19,
       height 39, and recip2[19] = 54; the pillar hides part of the billboard. */
    sdepth = 3; side = 1; scol = 81; hw = 19; sh = 39; recip2w = 54;
    f = 3;
    while (f != 0) {
        col = 0;
        while (col != 160) {
            ang = (colmap[col] + heading) & 63;
            dx = cost[ang]; dy = sint[ang];
            x = px; y = py; found = 0; dist = 5; hx = px; hy = py;
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 1; hx = x; hy = y; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 2; hx = x; hy = y; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 3; hx = x; hy = y; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 4; hx = x; hy = y; } }
            hbuf[col] = htab[dist];
            dbuf[col] = dist;
            ubuf[col] = ((hx >> 1) + (hy >> 1)) & 7;   /* the texture column at the hit */
            col = col + 1;
        }
        s = 0;
        while (s != 160) {
            dx = s - scol + hw;
            twohw = hw + hw;
            left = dx & 32768;
            a = (dx - twohw) & 32768;
            covers = 0;
            if (left == 0) { if (a != 0) { covers = 1; } }
            d = dbuf[s];
            near = (sdepth - d) & 32768;
            sbh = 0;
            if (covers != 0) { if (near != 0) { sbh = sh; } }
            su = 0;
            if (covers != 0) { su = (dx * recip2w) >> 8; }
            sbuf[s] = sbh;
            stex[s] = su;
            s = s + 1;
        }
        p = 16000;
        while (p != 0) {
            t = in_read();
            pcol = t & 255;
            prow = t >> 8;
            h = hbuf[pcol];
            d = dbuf[pcol];
            u = ubuf[pcol];
            top = 50 - h;
            bot = 50 + h;
            v = ((prow - top) * recip2[h]) >> 8;         /* the texture row: 0..7 down the wall */
            dark = (d - 3) & 32768;                       /* near walls (d < 3) use the bright bank */
            c = texture[((v & 7) << 3) | u];
            if (dark == 0) { c = texture[64 | ((v & 7) << 3) | u]; }
            a = (prow - top) & 32768;
            if (a != 0) { c = 1; }
            b = (bot - prow) & 32768;
            if (b != 0) { c = 2; }
            sbh = sbuf[pcol];
            su = stex[pcol];
            stop = 50 - (sbh >> 1);
            sbot = stop + sbh;
            a = (prow - stop) & 32768;
            b = (prow - sbot) & 32768;
            sv = 0;
            if (sbh != 0) { if (a == 0) { if (b != 0) { sv = ((prow - stop) * recip2[sbh]) >> 6; } } }
            /* Rounded reciprocals can produce texture row 16 at the final scaled pixel. */
            a = (sv - 16) & 32768;
            if (a == 0) { sv = 15; }
            spix = sprite[(sv << 3) | su];
            if (sbh != 0) { if (a == 0) { if (b != 0) { if (spix != 0) { c = spix; } } } }
            out_pixel(c);
            p = p - 1;
        }
        in = in_read();
        turn = in & 63;
        heading = (heading + turn) & 63;
        fwd = (in >> 8) & 3;
        nx = px; ny = py;
        if (fwd == 1) { nx = px + cost[heading]; ny = py + sint[heading]; }
        if (fwd == 2) { nx = px - cost[heading]; ny = py - sint[heading]; }
        cell = grid[((ny >> 4) << 4) | (nx >> 4)];   /* collision: a wall cell stops the move (Doom's p_map) */
        if (cell == 0) { px = nx; py = ny; }
        /* P_Move-shaped chase: choose the larger axis, take 8 Q4.4 units, and reject walls
           and the player cell.  tx and ty are loop-carried frame state. */
        tdx = tx - px; xneg = tdx & 32768;
        if (xneg != 0) { tdx = 0 - tdx; }
        tdy = ty - py; yneg = tdy & 32768;
        if (yneg != 0) { tdy = 0 - tdy; }
        bigger = (tdx - tdy) & 32768;
        nx = tx; ny = ty;
        if (bigger == 0) { if (xneg != 0) { nx = tx + 8; } else { nx = tx - 8; } }
        if (bigger != 0) { if (yneg != 0) { ny = ty + 8; } else { ny = ty - 8; } }
        tcell = grid[((ny >> 4) << 4) | (nx >> 4)];
        pcell = ((py >> 4) << 4) | (px >> 4);
        if (tcell == 0) { if ((((ny >> 4) << 4) | (nx >> 4)) != pcell) { tx = nx; ty = ny; } }
        /* These self reads tell the kernel compiler that the projected values are frame state. */
        sdepth = sdepth; scol = scol; hw = hw; sh = sh; recip2w = recip2w;
        relx = tx - px; rely = ty - py;
        camang = (heading + 10) & 63;
        cc = cost[camang]; ss = sint[camang];
        depth = ((relx * cc) + (rely * ss)) >> 4;
        sdepth = depth >> 4;
        if (sdepth == 0) { sdepth = 1; }
        sraw = ((0 - relx) * ss) + (rely * cc);
        sign = sraw & 32768;
        side = sraw;
        if (sign != 0) { side = 0 - sraw; }
        side = side >> 4;
        off = (side * recip2[sdepth]) >> 8;
        scol = 80 + off;
        if (sign != 0) { scol = 80 - off; }
        sh = (120 * recip2[sdepth]) >> 10;
        hw = (60 * recip2[sdepth]) >> 10;
        if (hw == 0) { hw = 1; }
        recip2w = recip2[hw];
        f = f - 1;
    }
    return 0;
}
