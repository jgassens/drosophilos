/* The world update with fresh input every tick (Stage D shape, kernel form): the same
   player / monster / walls / health rules as tick.c, but the velocity is read at the top of
   each tick, so a resident kernel takes it as its token stream and carries px, mx and health
   across ticks as state. Eight ticks; two pixel records per tick (player and monster). */
static u8 px = 20, mx = 90, health = 100, vel, dist, i, contact;

static void p_move(void) {
    px = px + vel;
    if ((px & 128) != 0) { px = 0; }
    contact = px ^ 200;
    if (contact == 0) { px = 199; }
}

static void m_move(void) {
    dist = mx - px;
    if ((dist & 128) != 0) { mx = mx + 2; } else { mx = mx - 2; }
    contact = mx ^ px;
    if (contact == 0) { health = health - 10; }
}

int main(void) {
    i = 8;
    while (i != 0) {
        vel = in_read();
        p_move();
        m_move();
        out_pixel(px);
        out_pixel(mx);
        i = i - 1;
    }
    out_pixel(health);
    return 0;
}
