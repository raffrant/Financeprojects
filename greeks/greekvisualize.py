import streamlit as st
import math
from dataclasses import dataclass
from scipy.stats import norm


# ===========================
# Core Black-Scholes "Hamiltonian"
# ===========================

def bs_price(S, K, T, r, sigma, q=0.0, call=True):
    """Black-Scholes price: think of this as E[payoff] under risk-neutral drift."""
    if T <= 0:
        if call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


# ===========================
# Physics-style finite differences
# ===========================

@dataclass
class BSPhysicsGreeks:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

    def price(self, call=True):
        return bs_price(self.S, self.K, self.T, self.r, self.sigma, self.q, call)

    def _central_diff(self, var: str, h: float, call=True):
        """Generic central derivative: (f(x+h)-f(x-h))/(2h)."""
        orig = getattr(self, var)

        setattr(self, var, orig + h)
        p_up = self.price(call)

        setattr(self, var, orig - h)
        p_down = self.price(call)

        setattr(self, var, orig)  # restore

        return (p_up - p_down) / (2.0 * h)

    def _second_central_diff_S(self, h: float, call=True):
        """Second derivative wrt S: (f(S+h)-2f(S)+f(S-h))/h^2."""
        orig_S = self.S

        self.S = orig_S + h
        p_up = self.price(call)

        self.S = orig_S
        p_mid = self.price(call)

        self.S = orig_S - h
        p_down = self.price(call)

        self.S = orig_S
        return (p_up - 2.0 * p_mid + p_down) / (h * h)

    # ---- Greeks as "responses" ----


    def delta(self, call=True):
        return self._central_diff("S", 1e-3, call)

    def gamma(self, call=True):
        return self._second_central_diff_S(1e-3, call)

    def vega(self, call=True):
        return self._central_diff("sigma", 1e-3, call)

    def theta(self, call=True):
        return self._central_diff("T", 1e-3, call)

    def rho(self, call=True):
        return self._central_diff("r", 1e-3, call)


# ===========================
# Streamlit UI
# ===========================

st.title("Physics-Inspired Black-Scholes Greeks")

st.write(
    """
Treat the option price as an **observable** and each Greek as the response to
a small perturbation of one parameter (S, σ, T, r), computed via central finite differences.
"""
)

col1, col2 = st.columns(2)

with col1:
    S = st.number_input("Spot Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=0.01, step=0.01)

with col2:
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.0001, format="%.4f")
    sigma = st.number_input("Volatility (σ)", value=0.20, min_value=0.001, step=0.001,format="%.3f")
    q = st.number_input("Dividend Yield (q)", value=0.0, step=0.001,format="%.3f")

option_type = st.selectbox("Option Type", ["Call", "Put"])
call = option_type == "Call"

# Build model
model = BSPhysicsGreeks(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

# Compute
price = model.price(call=call)
delta = model.delta(call=call)
gamma = model.gamma(call=call)
vega = model.vega(call=call)
theta = model.theta(call=call)
rho = model.rho(call=call)

st.subheader("Results")

st.write(f"### Option Price: {price:.6f}")
st.write("### Greeks")

colA, colB = st.columns(2)
with colA:
    st.write(f"**Delta:** {delta:.6f}")
    st.write(f"**Gamma:** {gamma:.6f}")
    st.write(f"**Vega:** {vega:.6f}")
with colB:
    st.write(f"**Theta:** {theta:.6f}")
    st.write(f"**Rho:** {rho:.6f}")

st.write(
    "All Greeks are computed as central finite-difference derivatives of the price, "
    "analogous to numerical response functions in physics simulations."
)
