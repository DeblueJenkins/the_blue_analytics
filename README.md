# This is an (Asian) option pricing library.


### Monte-Carlo simulation using Geometric Brownian Motion

There are currently four simulation methods implemented:

* Closed-form solution
* Euler-Maruyama
* Milstein
* Runge-Kutta

They are represented by the following equations:

$$
S_{t+\delta t} = S_{t}e^{(r - \frac{\sigma^{2}}{2})\delta t + \sigma\phi\\sqrt{delta t}}
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma \phi \sqrt{\delta t})
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma\phi\sqrt{\delta t} + \frac{1}{2} \sigma^{2}((\phi\sqrt{\delta t})^{2}) - \delta t)
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma \phi \sqrt{\delta t}) + \frac{1}{2}(\sigma(\hat{S} - S_{t}))((\phi\sqrt{\delta t})^{2} - \delta t)\frac{1}{\sqrt{\delta t}}
$$

where, $\hat{S} = S_{t}(1 + r\delta t + \sigma\sqrt{\delta t})$, while the Brownian variable is represented by $\phi\sqrt{\delta t}$, $\phi$ being a standard normal variable. 


The theoretical mean and variance of a Geometric Brownian motion are represented by Equations \ref{eq:theomean} and \ref{eq:theovar}. 

$$
\mathbb{E}[S] = S_{0}e^{rT}
$$

$$
\mathbb{V}[S] = S_{0}^{2}e^{2rT}(e^{\sigma^{2}T} - 1)
$$


### Closed-form solutions and approximations
 

#### Continuous Geometric

For the continuous geometric Asian option, the library uses the Kemna and Vorst (1990) solution.

$$
C = e^{-(b_{A}-r)T}SN(d1) - e^{-rT}N(d2)
$$

$$
P = e^{-rT}KN(-d2) - e^{-(b_{A}-r)T}SN(-d1)
$$

$$
d_{1} = \frac{ln(S/K)+(b_{A}+\sigma_{A}^{2}/2)T}{\sigma_{A}\sqrt{T}}\\
d_{2} = d_{1} - \sigma_{A}\sqrt{T}
$$

where, $b_{A}$ is the adjusted cost-of-carry and $\sigma_{A}$ is the adjusted volatility as given by:

$$
\sigma_{A} = \frac{\sigma}{\sqrt{3}}\\
b_{A} = \frac{1}{2}\bigg( b - \frac{\sigma^{2}}{6}\bigg)\\ 
$$

#### Discrete Geometric

For the discrete geometric Asian option there exists a closed-form solution. These can be computed using the Black-Scholes formula, however, the volatility needs to be adjusted. There are various ways to do this (see, Haug 2006), in which the vanilla Black-Scholes volatilities are combined in a piecewise manner. Alternatively, we can use a simpler modification:

$$
C = e^{-rT}SN(d1) - e^{-rT}N(d2)
$$

$$
P = e^{-rT}KN(-d2) - e^{-rT}SN(-d1)
$$

$$
d1 = \frac{ln(S/K) + b_{G} + \sigma_{G}^{2}/2}{\sigma_{G}}\\
d2 = d1 - \sigma_{G}
$$

where, again, $b_{G}$ and $\sigma_{G}$ are the adjusted cost of carry and volatility, given by:

$$
b_{G} = \frac{\sigma^{2}_{G}}{2 + (r-\frac{\sigma^{2}}{2})(t_{1}+\frac{T-t_{1}}{2})} \\
\sigma_{G} = \sqrt{\sigma^{2}(t_{1}+(T-t_{1})\frac{2n-1}{6n})}
$$

#### Continuous and Discrete Arithmetic

For the continuous geometric Asian option, the library uses the Turnbull and Wakeman (1991) approximations for an arithmetic average Asian option. 

$$
C \approx Se^{b_{A} - r}TN(d_{1}) - Ke^{-rT}N(d_{2})
$$

$$
P \approx Ke^{-rT}N(-d_{2}) - Se^{b_{A}-r}TN(-d_{1})
$$

$$
d_{1} = \frac{ln(S/K)+(b_{A}+\sigma_{A}^{2}/2)T}{\sigma_{A}\sqrt{T}}\\
d_{2} = d_{1} - \sigma_{A}\sqrt{T}
$$

where, $b_{A}$ and $\sigma_{A}$ are the adjusted cost of carry and volatility, and $b$ is the regular cost of carry, which in the simplest case of a non-dividend paying stock is simply the risk-free rate $b=r$:

$$
\sigma_{A} = \sqrt{\frac{ln(M_{2})}{T} - 2b_{A}} \\
b_{A} = \frac{M_{1}}{T} \\
$$

The exact first and second moments, $M_{1}$ and $M_{2}$ are given by:

$$
M_{1} = \frac{e^{bT} - e^{bt_{1}}}{b(T-t_{1})} \\
M_{2} = \frac{2e^{(2b+\sigma^{2})T}}{(b+\sigma^{2})(2b+\sigma^{2})(T-t_{1})^{2}} + \frac{2e^{(2b+\sigma^{2}})t_{1}}{b(T-t_{1})^{2}}\bigg(\frac{1}{2b+\sigma^{2}} - \frac{e^{b(T-t_{1})}}{b+\sigma^{2}}\bigg) \\
$$
	
where, $t_{1}$ is the time to the beginning of the averaging period. In our case, $t_{1}=0$. 

