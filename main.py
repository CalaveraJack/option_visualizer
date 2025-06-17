import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrow
from matplotlib.widgets import Slider

class Option:
    def __init__(self, strike, premium, opt_type, style, maturity_date, purchase_date,
                 asianing_dates=None, asian_average_spot = None, barrier=None, barrier_type=None,
                 barrier_monitoring=None, rebate=0):
        self.strike = strike
        self.premium = premium
        self.asian_average_spot = None
        self.opt_type = opt_type.lower()
        self.style = style.lower()
        self.maturity_date = maturity_date
        self.purchase_date = purchase_date
        self.asianing_dates = asianing_dates if asianing_dates is not None else []
    
        # Barrier-related
        self.barrier = barrier
        self.barrier_type = barrier_type  # e.g. 'kiko', 'up-in', 'down-out'
        self.barrier_monitoring = barrier_monitoring if barrier_monitoring is not None else []
        self.rebate = rebate

    def check_barrier(self, spot_history):
        if self.barrier is None or self.barrier_type is None:
            return "active"
    
        barrier_hit = any(
            (s >= self.barrier if "up" in self.barrier_type else s <= self.barrier)
            for s in spot_history
        )
    
        if self.barrier_type == "kiko":
            return "knockout" if barrier_hit else "inactive"
        elif "in" in self.barrier_type:
            return "active" if barrier_hit else "inactive"
        elif "out" in self.barrier_type:
            return "knockout" if barrier_hit else "active"
        return "active"


    def payoff(self, spot, time, avg_spot=None):
        if time < self.purchase_date:
            return np.zeros_like(spot)

        is_maturity = np.isclose(time, self.maturity_date)

        if self.style == 'asian':
            effective_avg_spot = avg_spot if avg_spot is not None else self.asian_average_spot
            if effective_avg_spot is not None:
                if self.opt_type == 'call':
                    return (np.maximum(effective_avg_spot - self.strike, 0) - self.premium) * np.ones_like(spot)
                elif self.opt_type == 'put':
                    return (np.maximum(self.strike - effective_avg_spot, 0) - self.premium) * np.ones_like(spot)


        # For non-Asian or when avg_spot is None
        if not is_maturity:
            if self.style in ['european', 'asian']:
                return np.full_like(spot, -self.premium)
            elif self.style == 'american':
                if self.opt_type == 'call':
                    return np.maximum(spot - self.strike, 0) - self.premium
                elif self.opt_type == 'put':
                    return np.maximum(self.strike - spot, 0) - self.premium

        if self.opt_type == 'call' and self.style != 'asian':
            return np.maximum(spot - self.strike, 0) - self.premium
        elif self.opt_type == 'put' and self.style != 'asian':
            return np.maximum(self.strike - spot, 0) - self.premium

    def calculate_pl_surface(self, spot_range, time_range, avg_spot_override=None):
        S, T = np.meshgrid(spot_range, time_range)
        PnL = np.zeros_like(S)

        for i, t in enumerate(time_range):
            if self.style == 'asian':
                # Use externally overridden average spot (e.g., from slider), if provided
                if avg_spot_override is not None:
                    avg_spot = avg_spot_override
                else:
                    # Compute average from path history approximation (based on time)
                    relevant_dates = [d for d in self.asianing_dates if d <= t]
                    avg_spot = np.mean(S[i, :]) if relevant_dates else None

                PnL[i, :] = self.payoff(S[i, :], t, avg_spot=avg_spot)
            else:
                PnL[i, :] = self.payoff(S[i, :], t)

        return S, T, PnL

    def __add__(self, other):
        if isinstance(other, Option):
            return Portfolio([self, other])
        elif isinstance(other, Portfolio):
            return Portfolio([self] + other.products)
        else:
            raise TypeError("Can only add Option or Portfolio to Option")


class Portfolio:
    def __init__(self, products):
        self.products = products

    def __add__(self, other):
        if isinstance(other, Option):
            return Portfolio(self.products + [other])
        elif isinstance(other, Portfolio):
            return Portfolio(self.products + other.products)
        else:
            raise TypeError("Can only add Option or Portfolio to Portfolio")


    def calculate_pl_surface(self, spot_range, time_range):
        S, T = np.meshgrid(spot_range, time_range)
        Total_PnL = np.zeros_like(S)
        for product in self.products:
            _, _, PnL = product.calculate_pl_surface(spot_range, time_range)
            Total_PnL += PnL
        return S, T, Total_PnL


def plot_3d_surface(X, Y, Z, title="P/L Surface", show_underlying="none", portfolio_like=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)

    if show_underlying in ("long", "both"):
        Z_long = X.copy()
        ax.plot_surface(X, Y, Z_long, alpha=0.2, color='green')
    if show_underlying in ("short", "both"):
        Z_short = -X.copy()
        ax.plot_surface(X, Y, Z_short, alpha=0.2, color='red')

    # Add barrier arrows if portfolio info is passed
    def draw_barrier_arrows(ax, time_range, barrier, kind):
        arrow_z = 0
        for t in np.linspace(time_range[0], time_range[-1], 10):
            if kind == 'knock-in':
                ax.quiver(barrier, t, arrow_z, 1, 0, 0, length=2, color='green', arrow_length_ratio=0.2)
            elif kind == 'knock-out':
                ax.quiver(barrier, t, arrow_z, -1, 0, 0, length=2, color='red', arrow_length_ratio=0.2)

    if portfolio_like is not None:
        for opt in getattr(portfolio_like, "products", [portfolio_like]):
            if opt.barrier is not None:
                kind = 'knock-in' if 'in' in opt.barrier_type else 'knock-out'
                draw_barrier_arrows(ax, Y[:, 0], opt.barrier, kind)

    # Interactive rotation with arrow keys
    elev, azim = 30, 135
    ax.view_init(elev=elev, azim=azim)

    def on_key(event):
        nonlocal elev, azim
        if event.key == 'left':
            azim -= 5
        elif event.key == 'right':
            azim += 5
        elif event.key == 'up':
            elev += 5
        elif event.key == 'down':
            elev -= 5
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Time")
    ax.set_zlabel("Profit / Loss")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_portfolio(portfolio_like, spot_range, time_range, show_underlying="none"):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if isinstance(portfolio_like, Option):
        portfolio = Portfolio([portfolio_like])
    else:
        portfolio = portfolio_like

    maturities = set(opt.maturity_date for opt in portfolio.products)
    has_american = any(opt.style == 'american' for opt in portfolio.products)
    has_varied_maturities = len(maturities) > 1
    asian_opts = [opt for opt in portfolio.products if opt.style == 'asian']

    # If 3D is required
    if has_american or has_varied_maturities:
        X, Y, Z = portfolio.calculate_pl_surface(spot_range, time_range)
        plot_3d_surface(X, Y, Z, title="Portfolio P/L Surface (3D)", show_underlying=show_underlying, portfolio_like=portfolio)
        return

    # -----------------------------
    # 2D Maturity Plot with Sliders
    # -----------------------------
    maturity_time = list(maturities)[0]

    # Init average spots if needed
    for asian_opt in asian_opts:
        if asian_opt.asian_average_spot is None:
            asian_opt.asian_average_spot = (spot_range[0] + spot_range[-1]) / 2

    # PnL computation at maturity
    def compute_total_pnl():
        total = np.zeros_like(spot_range)
        for opt in portfolio.products:
            total += opt.payoff(spot_range, maturity_time)
        return total

    initial_pnl = compute_total_pnl()

    # Set up figure and line plot
    n_sliders = len(asian_opts)
    slider_height = 0.04
    total_slider_height = slider_height * n_sliders
    fig_height = 5 + 3 if n_sliders else 5
    fig, ax = plt.subplots(figsize=(10, fig_height))
    plt.subplots_adjust(bottom=0.25 + total_slider_height)

    line, = ax.plot(spot_range, initial_pnl, label='Portfolio P/L')

    # Underlying
    if show_underlying in ("long", "both"):
        ax.plot(spot_range, spot_range, '--', label='Underlying (long)', color='green')
    if show_underlying in ("short", "both"):
        ax.plot(spot_range, -spot_range, '--', label='Underlying (short)', color='red')

    # Barriers
    for opt in portfolio.products:
        if opt.barrier is not None:
            kind = 'knock-in' if 'in' in opt.barrier_type else 'knock-out'
            barrier = opt.barrier
            ax.axvline(barrier, color='purple', linestyle='--', linewidth=1)
            arrow_y = ax.get_ylim()[1] * 0.8
            arrow_length = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
            if kind == 'knock-in':
                ax.annotate('', xy=(barrier + arrow_length, arrow_y), xytext=(barrier, arrow_y),
                            arrowprops=dict(arrowstyle="->", color="green", lw=2))
                ax.text(barrier + arrow_length, arrow_y * 1.02, 'Knock-In', color='green', fontsize=8)
            elif kind == 'knock-out':
                ax.annotate('', xy=(barrier - arrow_length, arrow_y), xytext=(barrier, arrow_y),
                            arrowprops=dict(arrowstyle="->", color="red", lw=2))
                ax.text(barrier - 2 * arrow_length, arrow_y * 1.02, 'Knock-Out', color='red', fontsize=8)

    ax.set_title("Portfolio P/L at Maturity (2D)")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Profit / Loss")
    ax.legend()
    ax.grid(True)

    # Add sliders for each Asian option
    sliders = []
    for i, asian_opt in enumerate(asian_opts):
        ax_slider = plt.axes([0.2, 0.2 - i * slider_height, 0.65, 0.03])
        slider = Slider(ax_slider, f'Asian Avg Spot {i + 1}', spot_range[0], spot_range[-1], valinit=asian_opt.asian_average_spot)
        sliders.append((slider, asian_opt))

        def make_update(opt):
            def update(val):
                opt.asian_average_spot = val
                new_pnl = compute_total_pnl()
                line.set_ydata(new_pnl)
                fig.canvas.draw_idle()
            return update

        slider.on_changed(make_update(asian_opt))

    plt.show()


# Example usage
if __name__ == "__main__":
    # Time and spot axis
    time_points = np.linspace(0, 1, 50)
    spot_prices = np.linspace(0, 50, 100)
    
    # 1. European Call
    call = Option(
        strike=20,
        premium=1,
        opt_type='call',
        style='european',
        maturity_date=1.0,
        purchase_date=0.2
    )

    # 2. American Put
    put = Option(
        strike=20,
        premium=1,
        opt_type='put',
        style='american',
        maturity_date=1.0,
        purchase_date=0.0
    )

    # 3. Asian Put #1
    asian1 = Option(
        strike=20,
        premium=1,
        opt_type='put',
        style='asian',
        maturity_date=1.0,
        purchase_date=0.3,
        asianing_dates=np.linspace(0.5, 1.0, 5),
        asian_average_spot=25  # Initial average value (can be adjusted by slider)
    )

    # 4. Asian Call #2
    asian2 = Option(
        strike=15,
        premium=1,
        opt_type='call',
        style='asian',
        maturity_date=1.0,
        purchase_date=0.4,
        asianing_dates=np.linspace(0.6, 1.0, 4),
        asian_average_spot=18  # Another one with a separate slider
    )

    # Combine into portfolio (uncomment call/put to test interaction)
    portfolio = asian1 + asian2  # + call + put

    # Visualize
    visualize_portfolio(portfolio, spot_prices, time_points, show_underlying="both")
