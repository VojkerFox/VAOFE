import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import os

def generate_setup_chart(symbol, rates, target_level, pivot_price, direction, output_path="setup_snapshot.png", tp1=None, tp2=None):
    """
    Piirtää alkuperäisen Break & Retest (B&R) graafin ja tallentaa sen kuvana.
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


def generate_liquidity_sweep_chart(symbol, rates, bsl_level, ssl_level, sweep_index=None, entry_price=None, direction=None, output_path="liquidity_sweep.png"):
    """
    Piirtää puhtaan likviditeettigraafin, merkitsee ansa-alueet (BSL/SSL) ja iskee nuolen kaupan kohdalle.
    """
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle='', y_on_right=True)
    
    hlines_vals = [bsl_level, ssl_level]
    hlines_colors = ['#ef4444', '#22c55e'] # Punainen (BSL) ja Vihreä (SSL)
    hlines_styles = [':', ':']
    hlines = dict(hlines=hlines_vals, colors=hlines_colors, linestyle=hlines_styles, linewidths=2, alpha=0.9)
    
    apds = []
    
    if sweep_index is not None and entry_price is not None and direction is not None:
        marker_series = pd.Series(index=df.index, dtype=float)
        actual_idx = sweep_index if sweep_index >= 0 else len(df) + sweep_index
        
        if 0 <= actual_idx < len(df):
            marker_series.iloc[actual_idx] = entry_price
            marker_shape = '^' if direction == 'BULL' else 'v'
            marker_color = '#22c55e' if direction == 'BULL' else '#ef4444'
            
            apds.append(mpf.make_addplot(marker_series, type='scatter', markersize=250, 
                                        marker=marker_shape, color=marker_color, alpha=1.0))
    
    title_text = f"{symbol} M15/H1 - LIQUIDITY SWEEP TRACKER"
    fig, axlist = mpf.plot(df, type='candle', style=style, 
                           title=title_text,
                           hlines=hlines,
                           addplot=apds if apds else None,
                           returnfig=True, figsize=(12, 7))
    
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path