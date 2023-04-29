# the_blue_analytics

This is a Asian option pricing library.

There are currently four simulation methods implemented:

* 

Simulation of GBM can be made in various ways. In this exercise, I have used the suggested methods (Closed-form solution, Euler-Maruyama, Milstein) as well as the Runge-Kutta method. Equations \ref{eq:closed} through \ref{eq:runge} represent the discretization schemes, respectively.  

$$
S_{t+\delta t} = S_{t}e^{(r - \frac{\sigma^{2}}{2})\delta t + \sigma\phi\delta t}
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma \phi \sqrt{\delta t})
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma\phi\sqrt{\delta t} + \frac{1}{2} \sigma^{2}((\phi\sqrt{\delta t})^{2}) - \delta t)
$$

$$
S_{t+\delta t} = S_{t}(1 + r\delta t + \sigma \phi \sqrt{\delta t}) + \frac{1}{2}(\sigma(\hat{S} - S_{t}))((\phi\delta t)^{2} - \delta t)\frac{1}{\sqrt{\delta t}}
$$

where, $\hat{S} = S_{t}(1 + r\delta t + \sigma\sqrt{\delta t})$, while the Brownian variable is represented by $\phi\delta t$, $\phi$ being a standard normal variable. 


The theoretical mean and variance of a Geometric Brownian motion are represented by Equations \ref{eq:theomean} and \ref{eq:theovar}. 

$$
\mathbb{E}[S] = S_{0}e^{rT}
$$

$$
\mathbb{V}[S] = S_{0}^{2}e^{2rT}(e^{\sigma^{2}T} - 1)
$$


I will first introduce a few approximations and closed-form solutions for various specifications of an Asian option contract. This will help me to determine whether my Monte-Carlo simulation are precise. 

\subsection{Continuous Geometric}

For the continuous geometric Asian option, I opted to use the Kemna and Vorst (1990) solution (Equations \ref{eq:cont_geo_call} and \ref{eq:cont_geo_put}).

\begin{equation}
C = e^{-(b_{A}-r)T}SN(d1) - e^{-rT}N(d2)
\label{eq:cont_geo_call}
\end{equation}

\begin{equation}
P = e^{-rT}KN(-d2) - e^{-(b_{A}-r)T}SN(-d1)
\label{eq:cont_geo_put}
\end{equation}

\begin{align*}
d_{1} &= \frac{ln(S/K)+(b_{A}+\sigma_{A}^{2}/2)T}{\sigma_{A}\sqrt{T}}\\
d_{2} &= d_{1} - \sigma_{A}\sqrt{T}
\end{align*}

where, $b_{A}$ is the adjusted cost-of-carry and $\sigma_{A}$ is the adjusted volatility as given by:

\begin{align*}
\sigma_{A} &= \frac{\sigma}{\sqrt{3}}\\
b_{A} &= \frac{1}{2}\bigg( b - \frac{\sigma^{2}}{6}\bigg)\\ 
\end{align*}

Table \ref{tab:cont_geo} displays the results for the given example in the instructions.

\begin{table}[h!]
\centering
\begin{tabular}{lll}
\toprule
 & approx/closed & monte carlo \\
\midrule
call & 5.5468 & 5.5419 \\
put & 3.4633 & 3.4490 \\
\bottomrule
\end{tabular}
\caption{\label{tab:cont_geo} Prices of puts and calls at $K=100$, $S=100$, $r=0.05$, $\sigma=0.2$, $T=1$ for Geometric continuous averaging Asian option. Number of steps is 1000, number of simulations is 30,000}
\end{table}

Indeed, as shown by Figure \ref{fig:cont_geo_convergence}, as I increase the number of simulations, the Monte-Carlo price converges to the closed-form solution.

\begin{figure}
\centering
\includegraphics[scale=0.35]{convergence_asian_geometric_continuous.png}
\caption{\label{fig:cont_geo_convergence} Convergence of Monte-Carlo when using the Kemna and Vorst closed form solution.}
\end{figure} 

\subsection{Discrete Geometric}

For the discrete geometric Asian option there exists a closed-form solution. These can be computed using the Black-Scholes formula, however, the volatility needs to be adjusted. There are various ways to do this (see, Haug 2006), in which the vanilla Black-Scholes volatilities are combined in a piecewise manner. Alternatively, we can use a simpler modification:

\begin{equation}
C = e^{-rT}SN(d1) - e^{-rT}N(d2)
\label{eq:discrete_geo_call}
\end{equation}

\begin{equation}
P = e^{-rT}KN(-d2) - e^{-rT}SN(-d1)
\label{eq:discrete_geo_call}
\end{equation}

\begin{align*}
d1 &= \frac{ln(S/K) + b_{G} + \sigma_{G}^{2}/2}{\sigma_{G}}\\
d2 &= d1 - \sigma_{G}
\end{align*}

where, again, $b_{G}$ and $\sigma_{G}$ are the adjusted cost of carry and volatility, given by:

\begin{align*}
b_{G} &= \frac{\sigma^{2}_{G}}{2 + (r-\frac{\sigma^{2}}{2})(t_{1}+\frac{T-t_{1}}{2})} \\
\sigma_{G} &= \sqrt{\sigma^{2}(t_{1}+(T-t_{1})\frac{2n-1}{6n})}
\end{align*}

\begin{table}[h!]
\centering
\begin{tabular}{lll}
\toprule
 & approx/closed & monte carlo \\
\midrule
call & 5.4392 & 5.4690 \\
put & 3.3827 & 3.3232 \\
\bottomrule
\end{tabular}
\caption{\label{tab:discrete_geo} Prices of puts and calls at $K=100$, $S=100$, $r=0.05$, $\sigma=0.2$, $T=1$ for Geometric discrete averaging Asian option}
\end{table}

Figure \ref{fig:discrete_geo} displays the range of moneyness as before. I can see that the pricing using the closed form solution is now closer. 

\begin{figure}[h!]
\centering
\includegraphics[scale=0.35]{asian_arithmetic_discrete.png}
\caption{\label{fig:discrete_geo} Various moneyness for the arithmetic discrete aveargin Asian option using monte carlo and Black-Scholes formula with an adjusted volatility.}
\end{figure} 

It is worth pointing out an issue that I had when valuing the \textit{discrete} Asian options. I figured there are two ways to do that: (1) use the exact number of time-steps in the simulation as is the sampling frequency (resets) and take an average, or (2) simulate as many time-steps as desired, however sample in such away so that the sampling frequency is preserved. The former is self-explanatory. The latter means that the number of time-steps in the simulation \textit{must} be divisible with the sampling frequency (e.g., 1200 time steps in simulation vs 12 (monthly) sampling frequency). The sampling would then take place at the intervals $i_{0}, i_{1}, ... i_{n}$ where $n=12$. However, I could not succeed in this, so I opted for the former. Since the discretization schemes would be imprecise in this case (only 12 steps), I am forced to used to closed-form solution. 


\subsection{Continuous and Discrete Arithmetic}

Equations \ref{eq:turnbull_call} and \ref{eq:turnbull_put} represent the Turnbull and Wakeman (1991) approximations for an arithmetic average Asian option. 

\begin{equation}
C \approx Se^{b_{A} - r}TN(d_{1}) - Ke^{-rT}N(d_{2})
\label{eq:turnbull_call}
\end{equation}	

\begin{equation}
P \approx Ke^{-rT}N(-d_{2}) - Se^{b_{A}-r}TN(-d_{1})
\label{eq:turnbull_put}
\end{equation}

\begin{align*}
d_{1} &= \frac{ln(S/K)+(b_{A}+\sigma_{A}^{2}/2)T}{\sigma_{A}\sqrt{T}}\\
d_{2} &= d_{1} - \sigma_{A}\sqrt{T}
\end{align*}

where, $b_{A}$ and $\sigma_{A}$ are the adjusted cost of carry and volatility, and $b$ is the regular cost of carry, which in the simplest case of a non-dividend paying stock is simply the risk-free rate $b=r$:

\begin{align*}
\sigma_{A} &= \sqrt{\frac{ln(M_{2})}{T} - 2b_{A}} \\
b_{A} &= \frac{M_{1}}{T} \\
\end{align*}

The exact first and second moments, $M_{1}$ and $M_{2}$ are given by:

\begin{align*}
M_{1} &= \frac{e^{bT} - e^{bt_{1}}}{b(T-t_{1})} \\
M_{2} &= \frac{2e^{(2b+\sigma^{2})T}}{(b+\sigma^{2})(2b+\sigma^{2})(T-t_{1})^{2}} + \frac{2e^{(2b+\sigma^{2}})t_{1}}{b(T-t_{1})^{2}}\bigg(\frac{1}{2b+\sigma^{2}} - \frac{e^{b(T-t_{1})}}{b+\sigma^{2}}\bigg) \\
\end{align*}
	
where, $t_{1}$ is the time to the beginning of the averaging period. In our case, $t_{1}=0$. 

Table \ref{tab:discrete_ari} displays the results for an ATM put and call. Since the Turnbull and Wakemane's formula is just an approximation, I am inclined to believe that the Monte-Carlo simulation is more precise.

\begin{tabular}{llll}
\toprule
 & approx/closed & monte carlo (cont.) & monte carlo (discr.) \\
\midrule
call & 5.7828 & 5.7575 & 5.7049 \\ \\
put & 3.3646 & 3.3306 & 3.1976 \\\\
\bottomrule
\end{tabular}


For the discrete averaging, I repeated this exercise for different moneyness $[0.8, 0.9, ... 1.2]$ (in \% of $S_{0}$). Figure \ref{fig:discrete_ari} displays the results.

\begin{figure}[h!]
\centering
\includegraphics[scale=0.35]{asian_arithmetic_discrete.png}
\caption{\label{fig:discrete_ari} Various moneyness for the arithmetic discrete aveargin Asian option using monte carlo and Turnbull and Wakemane's approximation.}
\end{figure} 


\subsection{Fixed vs Floating Strike Price}

As a final exercise, I simulated and price a geometric continuous monitoring Asian option using both floating and fixed strike price. Table \ref{tab:fixed_floating} displays the results. I observe that the floating call is more expensive, while the floating put is cheaper. 

\begin{table}[h!]
\centering
\begin{tabular}{lll}
\toprule
 & monte carlo (fixed) & monte carlo (float) \\
\midrule
call & 5.5419 & 6.0364 \\\\
put & 3.4490 & 3.2693 \\\\
\bottomrule
\end{tabular}
\caption{\label{tab:fixed_floating} Geometric continuous Asian option using fixed and floating strike prices}
\end{table}
