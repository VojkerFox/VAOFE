import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import os

def generate_setup_chart(symbol, rates, target_level, pivot_price, direction, output_path="setup_snapshot.png", tp1=None, tp2=None):
    """
    Piirtää graafin ja tallentaa sen kuvana.
    rates: list of dicts (MT5 rates)
    target_level: BOS-taso (oranssi, SL)
    pivot_price: Entry-hinta (keltainen)
    tp1, tp2: 1:1 ja 1:2 Targetit
    """
    # 1. Muutetaan data DataFrameksi
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # 2. Määritetään visuaalinen tyyli
    style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle='', y_on_right=True)
    
    # 3. Luodaan perusviivat (SL ja Entry)
    hlines_vals = [target_level, pivot_price]
    hlines_colors = ['orange', 'yellow']
    hlines_styles = ['-.', '--']
    
    # Jos Targetit (TP1, TP2) on annettu (Vaihe 2), piirretään ne vihreinä
    if tp1 is not None and tp2 is not None:
        hlines_vals.extend([tp1, tp2])
        hlines_colors.extend(['#22c55e', '#16a34a']) # Vaaleanvihreä (1:1) ja Tummanvihreä (1:2)
        hlines_styles.extend(['--', '-.'])
        
    hlines = dict(hlines=hlines_vals, colors=hlines_colors, linestyle=hlines_styles, linewidths=1.5, alpha=0.8)
    
    # 4. Varjostetaan B&R Riskilaatikko (SL:n ja Entryn väli)
    # Käytetään vihreää jos ollaan ostamassa (Bull) ja punaista jos myymässä (Bear)
    fill_color = 'green' if direction == 'BULL' else 'red'
    fill_y1 = [target_level] * len(df)
    fill_y2 = [pivot_price] * len(df)
    
    # 5. Piirretään graafi varjostuksella ja viivoilla
    fig, axlist = mpf.plot(df, type='candle', style=style, 
                           title=f"{symbol} - VAOFE B&R Setup",
                           hlines=hlines,
                           fill_between=dict(y1=fill_y1, y2=fill_y2, color=fill_color, alpha=0.15),
                           returnfig=True, figsize=(10, 6))
    
    # Tallenna
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path