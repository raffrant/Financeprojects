
import streamlit as st
import math
from dataclasses import dataclass
from scipy.stats import norm

# ===========================
# Physics-Inspired Greeks Class
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
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q

        d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if call:
            return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        else:
            return K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)

    # Generic central-difference derivative
    def derivative(self, var: str, h: float = 1e-4, call=True):
        orig = getattr(self, var)

        setattr(self, var, orig + h)
        p_up = self.price(call)

        setattr(self, var, orig - h)
        p_down = self.price(call)

        setattr(self, var, orig)  # restore state

        return (p_up - p_down) / (2 * h)

    def delta(self, call=True):
        return self.derivative("S", call=call)

    def gamma(self, h=1e-3, call=True):
        orig_S = self.S

        self.S = orig_S + h
        p_up = self.price(call)
        self.S = orig_S
        p_mid = self.price(call)
        self.S = orig_S - h
        p_down = self.price(call)

        self.S = orig_S
        return (p_up - 2*p_mid + p_down) / (h*h)

    def vega(self, call=True):
        return self.derivative("sigma", call=call)

    def theta(self, call=True):
        return self.derivative("T", call=call)

    def rho(self, call=True):
        return self.derivative("r", call=call)


# ===========================
# Streamlit UI
# ===========================

st.title("Physics-Inspired Black-Scholes Greeks Calculator")

st.write(
    """
Compute Black-Scholes option prices and Greeks using a **finite-difference sensitivity approach**, similar to perturbation methods in physics simulations.
"""
)

col1, col2 = st.columns(2)

with col1:
    S = st.number_input("Spot Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (T, in years)", value=1.0, min_value=0.01, step=0.01)

with col2:
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.0001)
    sigma = st.number_input("Volatility (σ)", value=0.20, min_value=0.001, step=0.001)
    q = st.number_input("Dividend Yield (q)", value=0.0, step=0.001)

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

st.write("Greeks computed via central finite differences for stability and physical interpretability.")
