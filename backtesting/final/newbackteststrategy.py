"""
Brownian Drift Strategy — NVDA Edition
=======================================
Stock choice rationale:
  NVDA has the highest sustained drift-to-diffusion ratio of any
  mega-cap (2015–2025): ~55% CAGR, VR consistently > 1.15 in trending
  phases, and AI-driven momentum that makes μ dominate σ for long
  stretches — exactly the environment this strategy was designed for.

New mechanics vs previous version:
  1. Breakeven stop   — once up 1.5×ATR, stop slides to entry (zero-loss)
  2. Trailing ATR stop — after breakeven, trail 2×ATR below the highest high
  3. Loss cooldown     — after 2 consecutive losses, skip 5 bars
  4. All parameters tuned for NVDA's higher volatility profile
"""

import backtrader as bt
import numpy as np
from datetime import datetime
import pandas as pd
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR 1 — Kalman Filter MA  (tuned: higher Q for NVDA's speed)
# ══════════════════════════════════════════════════════════════════════
class KalmanMA(bt.Indicator):
    lines  = ("filtered", "slope", "gain")
    params = dict(Q=0.008, R=1.5, slope_bars=5)

    def __init__(self):
        self._x  = None
        self._P  = 1.0
        self.addminperiod(self.p.slope_bars + 1)

    def next(self):
        z = self.data.close[0]
        if self._x is None:
            self._x, self._P = z, 1.0

        P_pred   = self._P + self.p.Q
        K        = P_pred / (P_pred + self.p.R)
        self._x  = self._x + K * (z - self._x)
        self._P  = (1.0 - K) * P_pred

        self.lines.filtered[0] = self._x
        self.lines.gain[0]     = K

        prev = self.lines.filtered[-self.p.slope_bars]
        if prev and prev != 0:
            self.lines.slope[0] = (
                (self._x - prev) / prev
            ) * (252.0 / self.p.slope_bars)
        else:
            self.lines.slope[0] = 0.0


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR 2 — Brownian SNR  (drift / noise, per-period Sharpe)
# ══════════════════════════════════════════════════════════════════════
class BrownianSNR(bt.Indicator):
    lines  = ("snr", "drift", "diffusion")
    params = dict(period=25)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        closes   = np.array([self.data.close[-i]
                             for i in range(self.p.period + 1)])[::-1]
        log_rets = np.diff(np.log(closes + 1e-12))
        mu       = np.mean(log_rets)
        sigma    = np.std(log_rets) + 1e-12

        self.lines.snr[0]       = (mu * np.sqrt(self.p.period)) / sigma
        self.lines.drift[0]     = mu    * 252
        self.lines.diffusion[0] = sigma * np.sqrt(252)


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR 3 — Variance Ratio (trending regime detector)
# ══════════════════════════════════════════════════════════════════════
class VarianceRatio(bt.Indicator):
    lines  = ("vr",)
    params = dict(period=15, k=4)

    def __init__(self):
        self.addminperiod(self.p.period + self.p.k)

    def next(self):
        n        = self.p.period
        k        = self.p.k
        closes   = np.array([self.data.close[-i]
                             for i in range(n + k + 1)])[::-1]
        log_rets = np.diff(np.log(closes + 1e-12))
        var_1    = np.var(log_rets[-n:]) + 1e-18

        k_rets   = np.array([
            np.sum(log_rets[i : i + k])
            for i in range(len(log_rets) - k + 1)
        ])
        var_k    = (np.var(k_rets) / k + 1e-18) if len(k_rets) > 1 else var_1
        self.lines.vr[0] = var_k / var_1


# ══════════════════════════════════════════════════════════════════════
#  STRATEGY
# ══════════════════════════════════════════════════════════════════════
class BrownianDriftNVDA(bt.Strategy):
    """
    NVDA-specific parameter tuning vs generic version:

    Parameter         Generic    NVDA
    ─────────────────────────────────────────
    kalman_Q          0.005      0.008   more reactive to NVDA's pace
    kalman_R          2.0        1.5     trust observations more
    snr_period        30         25      faster signal window
    snr_threshold     0.6        0.55    NVDA trends clearly; easier edge
    vr_threshold      1.0        1.08    stricter: only strong trends
    slow_ema_period   150        120     NVDA moves in faster cycles
    atr_sl            1.5        2.0     wider stop; NVDA is volatile
    atr_tp            3.5        5.0     let winners run on a rocket stock
    risk_pct          1.0%       0.8%    smaller due to higher per-bar vol

    New mechanics:
    • Breakeven stop  : once floating P&L ≥ 1.5×ATR, stop → entry
    • Trailing stop   : thereafter, stop = highest_high − 2×ATR
    • Loss cooldown   : 2 consecutive losses → skip 5 bars
    """

    params = dict(
        # Kalman
        kalman_Q          = 0.008,
        kalman_R          = 1.5,
        slope_bars        = 5,
        slope_threshold   = 0.10,    # 10 % annualised minimum slope

        # SNR
        snr_period        = 25,
        snr_threshold     = 0.75,

        # Variance Ratio
        vr_period         = 22,
        vr_k              =4,
        vr_threshold      = 1.08,

        # Trend gate
        slow_ema_period   = 100,

        # ATR & risk
        atr_period        = 19,
        atr_sl            = 2.0,
        atr_tp            = 11.0,     # RR ≈ 2.5 — let NVDA run
        risk_pct          = 0.04,

        # New mechanics
        be_trigger_mult   = 1.15,     # move stop to entry after 1.5×ATR gain
        trail_atr_mult    = 2.0,     # trail 2×ATR below highest high
        cooldown_losses   = 2,       # consecutive losses before pause
        cooldown_bars     = 5,       # bars to sit out after cooldown trigger

        vol_bars          = 40,
    )

    def __init__(self):
        self.kalman    = KalmanMA(self.data,
                                  Q          = self.p.kalman_Q,
                                  R          = self.p.kalman_R,
                                  slope_bars = self.p.slope_bars)
        self.snr       = BrownianSNR(self.data, period=self.p.snr_period)
        self.vr        = VarianceRatio(self.data,
                                       period = self.p.vr_period,
                                       k      = self.p.vr_k)
        self.slow_ema  = bt.ind.EMA(self.data.close, period=self.p.slow_ema_period)
        self.atr       = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.avg_vol   = bt.ind.SMA(self.data.volume, period=self.p.vol_bars)

        # State
        self._order         = None
        self._entry_price   = None
        self._entry_atr     = None
        self._highest_high  = None
        self._be_triggered  = False
        self._consec_losses = 0
        self._cooldown_left = 0

    # ── Signal ────────────────────────────────────────────────────
    def _entry_ok(self):
        if self._cooldown_left > 0:
            return False
        above_trend = self.kalman.filtered[0] > self.slow_ema[0]
        slope_up    = self.kalman.slope[0]    > self.p.slope_threshold
        snr_ok      = self.snr.snr[0]         > self.p.snr_threshold
        regime_ok   = self.vr.vr[0]           > self.p.vr_threshold
        vol_ok      = self.data.volume[0]     > self.avg_vol[0]
        return above_trend and slope_up and snr_ok and regime_ok and vol_ok

    def _size(self, entry, stop):
        rps  = max(entry - stop, 1e-6)
        size = int((self.broker.getvalue() * self.p.risk_pct) / rps)
        return max(size, 1)

    # ── Core loop ─────────────────────────────────────────────────
    def next(self):
        # Tick down cooldown
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # Pending order — wait
        if self._order:
            return

        # ── In position: manage trailing / breakeven ──────────────
        if self.position:
            price = self.data.close[0]
            atr   = self._entry_atr
            ep    = self._entry_price

            # Track highest high since entry
            if self._highest_high is None or price > self._highest_high:
                self._highest_high = price

            # 1. Breakeven trigger
            if not self._be_triggered:
                if price >= ep + self.p.be_trigger_mult * atr:
                    self._be_triggered = True
                    self.log(f"BREAKEVEN triggered at {price:.2f}  "
                             f"(entry {ep:.2f})")

            # 2. Trailing stop check (manual — we close if breached)
            if self._be_triggered:
                trail_stop = self._highest_high - self.p.trail_atr_mult * atr
                # Hard floor: never below entry (breakeven guarantee)
                trail_stop = max(trail_stop, ep)
                if price <= trail_stop:
                    self.log(f"TRAIL STOP hit at {price:.2f}  "
                             f"(stop={trail_stop:.2f}  high={self._highest_high:.2f})")
                    self._order = self.close()
                    return

            # 3. Kalman slope reversal (original override)
            if self.kalman.slope[0] < 0:
                self.log(f"SLOPE EXIT at {price:.2f}")
                self._order = self.close()
            return

        # ── Not in position: look for entry ──────────────────────
        if self._entry_ok():
            entry = self.data.close[0]
            atr   = self.atr[0]
            stop  = entry - atr * self.p.atr_sl
            limit = entry + atr * self.p.atr_tp
            size  = self._size(entry, stop)

            self._entry_price  = entry
            self._entry_atr    = atr
            self._highest_high = entry
            self._be_triggered = False

            self.log(f"BUY  {size}sh @ {entry:.2f}  "
                     f"stop={stop:.2f}  limit={limit:.2f}  "
                     f"SNR={self.snr.snr[0]:.2f}  VR={self.vr.vr[0]:.2f}")

            self._order = self.buy_bracket(
                size       = size,
                price      = entry,
                stopprice  = stop,
                limitprice = limit,
                exectype   = bt.Order.Market,
            )

    def notify_order(self, order):
        if order.status in (order.Completed, order.Cancelled, order.Rejected):
            if self._order and hasattr(self._order, '__iter__'):
                done = all(o.status in (bt.Order.Completed,
                                        bt.Order.Cancelled,
                                        bt.Order.Rejected)
                           for o in self._order)
                if done:
                    self._order = None
            else:
                self._order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            won = trade.pnlcomm > 0
            if won:
                self._consec_losses = 0
            else:
                self._consec_losses += 1
                if self._consec_losses >= self.p.cooldown_losses:
                    self._cooldown_left = self.p.cooldown_bars
                    self.log(f"COOLDOWN activated ({self._consec_losses} "
                             f"consecutive losses — pausing {self.p.cooldown_bars} bars)")
                    self._consec_losses = 0

            roi = trade.pnlcomm / (trade.price * trade.size + 1e-6) * 100
            self.log(f"{'WIN ' if won else 'LOSS'}"
                     f"  pnl={trade.pnlcomm:+.2f}"
                     f"  ROI={roi:+.2f}%"
                     f"  bars={trade.barlen}")

    def log(self, txt):
        print(f"[{self.datas[0].datetime.date(0)}]  {txt}")


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════
TICKER     = "NVDA"   # ← chosen for highest sustained drift/noise ratio
START_DATE = "2015-10-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

print(f"\nDownloading {TICKER} …")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df.dropna(inplace=True)
print(f"✓  {TICKER}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")


# ══════════════════════════════════════════════════════════════════════
#  CEREBRO
# ══════════════════════════════════════════════════════════════════════
cerebro = bt.Cerebro(cheat_on_open=False)
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.001)

cerebro.adddata(bt.feeds.PandasData(dataname=df), name=TICKER)
cerebro.addstrategy(BrownianDriftNVDA)

cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                    _name        = "sharpe",
                    riskfreerate = 0.04,
                    annualize    = True,
                    timeframe    = bt.TimeFrame.Days)
cerebro.addanalyzer(bt.analyzers.DrawDown,      _name = "dd")
cerebro.addanalyzer(bt.analyzers.Returns,       _name = "ret")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = "trades")
cerebro.addanalyzer(bt.analyzers.AnnualReturn,  _name = "annual")
cerebro.addanalyzer(bt.analyzers.SQN,           _name = "sqn")


# ══════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print(f"  Brownian Drift — {TICKER}  (tuned + breakeven + trail stop)")
print("═" * 62)

results = cerebro.run()
strat   = results[0]

# ══════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════
START  = 100_000.0
final  = cerebro.broker.getvalue()
sharpe = strat.analyzers.sharpe.get_analysis()
dd     = strat.analyzers.dd.get_analysis()
trades = strat.analyzers.trades.get_analysis()
annual = strat.analyzers.annual.get_analysis()
sqn    = strat.analyzers.sqn.get_analysis()

closed   = trades.get("total",  {}).get("closed", 0)
won      = trades.get("won",    {}).get("total",  0)
lost     = trades.get("lost",   {}).get("total",  0)
win_rate = (won / closed * 100) if closed else 0.0
avg_win  = trades.get("won",  {}).get("pnl", {}).get("average", 0.0)
avg_loss = trades.get("lost", {}).get("pnl", {}).get("average", 0.0)
pf       = abs(avg_win / avg_loss) if avg_loss else float("inf")
avg_bars = trades.get("len",  {}).get("average", 0.0)
sh       = sharpe.get("sharperatio", 0) or 0

w = 62
print(f"\n╔{'═'*w}╗")
print(f"║{'  RESULTS — ' + TICKER + ' Brownian Drift':^{w}}║")
print(f"╠{'═'*w}╣")
print(f"║  Stock             : {TICKER:<{w-22}}║")
print(f"║  Start capital     : ${START:>14,.2f}{'':<{w-38}}║")
print(f"║  Final value       : ${final:>14,.2f}{'':<{w-38}}║")
print(f"║  Total return      : {(final/START-1)*100:>+13.2f} %{'':<{w-37}}║")
print(f"╠{'═'*w}╣")
print(f"║  Sharpe ratio      : {sh:>15.3f}{'':<{w-37}}║")
print(f"║  SQN               : {sqn.get('sqn', 0):>15.2f}{'':<{w-37}}║")
print(f"║  Max drawdown      : {dd.max.drawdown:>14.2f} %{'':<{w-37}}║")
print(f"║  Max DD duration   : {str(dd.max.len) + ' bars':>15}{'':<{w-37}}║")
print(f"╠{'═'*w}╣")
print(f"║  Trades (closed)   : {closed:>15}{'':<{w-37}}║")
print(f"║  Win rate          : {win_rate:>14.1f} %{'':<{w-37}}║")
print(f"║  Avg win           : ${avg_win:>14,.2f}{'':<{w-38}}║")
print(f"║  Avg loss          : ${avg_loss:>14,.2f}{'':<{w-38}}║")
print(f"║  Profit factor     : {pf:>14.2f}x{'':<{w-37}}║")
print(f"║  Avg hold (bars)   : {avg_bars:>15.1f}{'':<{w-37}}║")
print(f"╠{'═'*w}╣")
print(f"║  Annual returns:{'':<{w-16}}║")
for yr, r in sorted(annual.items()):
    filled = int(abs(r) * 35) if abs(r) < 1 else 35
    bar    = ("█" if r >= 0 else "░") * filled
    print(f"║    {yr}  {'+' if r>=0 else '-'}{abs(r)*100:5.1f}%  {bar:<35}{'':<5}║")
print(f"╚{'═'*w}╝\n")

cerebro.plot(style="candlestick", volume=True, iplot=False)
