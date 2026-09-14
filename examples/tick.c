/* A toy world-update tick in the shape of Doom's p_tick (Stage D shape): one player and one
   monster on a line of 8-bit positions inside walls at 0 and 200; the player moves by the
   input velocity and is clamped by the walls (collision); the monster steps toward the
   player; contact costs health. Three ticks per run; the "frame" is one pixel per entity. */
static u8 px = 20, mx = 90, health = 100, vel, dist, i, contact;

static void p_move(void) {
    px = px + vel;
    if ((px & 128) != 0) { px = 0; }         /* hit the west wall (wrapped negative) */
    contact = px ^ 200;
    if (contact == 0) { px = 199; }          /* hit the east wall */
}

static void m_move(void) {
    dist = mx - px;
    if ((dist & 128) != 0) { mx = mx + 2; } else { mx = mx - 2; }   /* step toward the player */
    contact = mx ^ px;
    if (contact == 0) { health = health - 10; }
}

int main(void) {
    vel = in_read();
    i = 3;
    while (i != 0) {
        p_move();
        m_move();
        out_pixel(px);
        out_pixel(mx);
        i = i - 1;
    }
    out_pixel(health);
    return 0;
}
