import os
import sys
import MetaTrader5 as mt5
import jax.numpy as jnp
from jax import jit, vmap, lax
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from walker_engine import SYMBOLS

# --- JAX-YDIN ---
STATE_IDLE = 0
STATE_ACTION = 1

@jit
def rdaas_fsm_step(carry, xs):
    # Puretaan 8 muuttujaa: (high, low, close, h1_res, h1_sup, h1_bull, h1_bear, is_trade)
    state, entry, sl, be_hit, max_box, cooldown, direction = carry
    high, low, close, h1_res, h1_sup, h1_bull, h1_bear, is_trade = xs
    
    cooldown = jnp.maximum(0, cooldown - 1)
    
    can_trade = (state == STATE_IDLE) & (cooldown == 0) & is_trade
    trigger_long = can_trade & h1_bull & (close > h1_res)
    trigger_short = can_trade & h1_bear & (close < h1_sup)

    state = jnp.where(trigger_long | trigger_short, STATE_ACTION, state)
    entry = jnp.where(trigger_long | trigger_short, close, entry)
    direction = jnp.where(trigger_long, 1.0, jnp.where(trigger_short, -1.0, direction))
    
    risk = jnp.maximum(jnp.abs(entry - jnp.where(trigger_long, h1_res, h1_sup)), close * 0.0015)
    sl = jnp.where(trigger_long, entry - risk, jnp.where(trigger_short, entry + risk, sl))
    max_box = jnp.where(trigger_long | trigger_short, 0.0, max_box)
    be_hit = jnp.where(trigger_long | trigger_short, 0.0, be_hit)

    is_active = (state == STATE_ACTION)
    profit = jnp.where(direction == 1.0, high - entry, entry - low)
    max_box = jnp.where(is_active, jnp.maximum(max_box, jnp.floor(profit / risk)), max_box)
    be_hit = jnp.where(is_active & (max_box >= 0.2), 1.0, be_hit)
    
    hit_sl = is_active & jnp.where(direction == 1.0, low <= jnp.where(be_hit, entry, sl), high >= jnp.where(be_hit, entry, sl))
    # Yksinkertaistettu palkkio
    payout = jnp.where(max_box < 1.0, -1.0, jnp.where(max_box == 1.0, 0.0, max_box - 1.0))
    
    state = jnp.where(hit_sl, STATE_IDLE, state)
    cooldown = jnp.where(hit_sl, 24, cooldown)
    return (state, entry, sl, be_hit, max_box, cooldown, direction), (hit_sl, payout, direction, entry, max_box)

vmap_rdaas = vmap(lambda c, x: lax.scan(rdaas_fsm_step, c, x))

def run_ytd_audit():
    if not mt5.initialize(): return
    curr, end = datetime(2026, 1, 1), datetime.now()
    total_r = 0.0
    
    while curr < end:
        nxt = (curr.replace(day=28) + timedelta(days=4)).replace(day=1)
        if nxt > end: nxt = end
        
        master = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, curr, nxt)
        if master is None or len(master) == 0: curr = nxt; continue
        m_times = [int(c['time']) for c in master]
        
        h_all, l_all, c_all, res_all, sup_all, bull_all, bear_all, win_all = [], [], [], [], [], [], [], []

        for sym in SYMBOLS:
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M15, curr, nxt)
            data = {int(c['time']): c for c in rates} if rates is not None else {}
            h, l, c = [], [], []
            last = rates[0] if rates is not None and len(rates) > 0 else {'high':0,'low':0,'close':0}
            for t in m_times:
                b = data.get(t, last)
                h.append(float(b['high'])); l.append(float(b['low'])); c.append(float(b['close']))
                last = b
            h_j, l_j, c_j = jnp.array(h), jnp.array(l), jnp.array(c)
            # Staattinen ikkuna JAX-yhteensopivuuden varmistamiseksi
            final_max, final_min = lax.fori_loop(96, len(c_j), lambda i, carry: (
                carry[0].at[i].set(jnp.max(lax.dynamic_slice(h_j, (i-96,), (96,)))),
                carry[1].at[i].set(jnp.min(lax.dynamic_slice(l_j, (i-96,), (96,))))
            ), (jnp.zeros_like(h_j), jnp.zeros_like(l_j)))
            
            h_all.append(h_j); l_all.append(l_j); c_all.append(c_j)
            res_all.append(final_max); sup_all.append(final_min)
            bull_all.append(c_j > final_max); bear_all.append(c_j < final_min)
            win_all.append(jnp.array([9 <= datetime.fromtimestamp(t).hour < 22 for t in m_times]))

        _, (sigs, profs, _, _, _) = vmap_rdaas(
            (jnp.zeros(len(SYMBOLS), int), jnp.zeros(len(SYMBOLS)), jnp.zeros(len(SYMBOLS)), jnp.zeros(len(SYMBOLS)), jnp.zeros(len(SYMBOLS)), jnp.zeros(len(SYMBOLS), int), jnp.zeros(len(SYMBOLS))),
            (jnp.stack(h_all), jnp.stack(l_all), jnp.stack(c_all), jnp.stack(res_all), jnp.stack(sup_all), jnp.stack(bull_all), jnp.stack(bear_all), jnp.stack(win_all))
        )
        
        for t in range(len(m_times)):
            for s in range(len(SYMBOLS)):
                if sigs[s, t]: total_r += float(profs[s, t])
        curr = nxt
        
    print(f"\n✅ YTD 2026 NETTOTUOTTO (0.2R BE): {total_r:+.2f} R")
    mt5.shutdown()

if __name__ == "__main__": run_ytd_audit()