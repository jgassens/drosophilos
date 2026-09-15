/* A Doom-like renderer with a moving player (Stage D + E1 shape): the tick kernel turns and
   moves the player from its input (bits 0-5 the turn, bits 8-9 forward / back), the column
   pass casts one ray per screen column through a 16 x 16 grid level in 4 steps of one cell
   (Q4.4 coordinates, trig tables), stores the projected wall height per column into a buffer,
   and the pixel pass paints ceiling, a textured wall (an 8 x 8 brick texture indexed by the
   hit position and the row within the wall, a dark bank beyond two cells), or floor by comparing
   the pixel's row with the column's height read back from the buffer. Level, trig, height and shade
   tables are compiled in by the asset compiler. Screen 160 x 100, the horizon at row 50. */
static u16 grid[256] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static u16 cost[64] = {16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534, 0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16};
static u16 sint[64] = {0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534};
static u16 htab[5] = {4, 40, 20, 13, 10};
static u16 shade[5] = {7, 3, 4, 4, 5};
static u16 colmap[160] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19};
static u16 hbuf[160];
static u16 dbuf[160];
static u16 ubuf[160];
static u16 texture[128] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 3, 3, 3, 3, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14, 14, 14, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14, 14, 14, 14, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7};   /* 8 x 8 bricks, a bright and a dark bank */
static u16 recip2[41] = {1024, 1024, 512, 341, 256, 205, 171, 146, 128, 114, 102, 93, 85, 79, 73, 68, 64, 60, 57, 54, 51, 49, 47, 45, 43, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 28, 27, 26, 26};
static u16 px, py, heading, in, turn, fwd, f, col, ang, dx, dy, x, y, cell, found, dist, h, nx, ny;
static u16 t, prow, pcol, top, bot, a, b, c, d, p, u, v, dark, hx, hy;

int main(void) {
    px = 128; py = 128; heading = 0;   /* the middle of the level, looking along +x (Q4.4) */
    f = 2;
    while (f != 0) {
        col = 0;
        while (col != 160) {
            ang = (colmap[col] + heading) & 63;
            dx = cost[ang]; dy = sint[ang];
            x = px; y = py; found = 0; dist = 0; hx = px; hy = py;
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
        f = f - 1;
    }
    return 0;
}
