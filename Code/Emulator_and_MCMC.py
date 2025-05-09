
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import trange


sns.set_theme()
params = {
    "font.family": "serif",
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "legend.fontsize": "medium"
}

plt.rcParams.update(params)
tab10 = sns.color_palette("tab10")


def compute_nll(y_model, y_obs, sigma):
    """
    Computing the Negative log likelihood.
    """
    return -np.sum(norm.logpdf(y_obs, loc=y_model, scale=sigma))



class EmulatorSampler:
    """

    
    """
    def __init__(self, priors_csv, truth_csv, runs_dir, sigma_nll=2.5, winsor_pct=95):
        """

        """
        # Load Priors data
        df_p = pd.read_csv(priors_csv, index_col=0)
        self.priors_df = df_p

        # Load truth data
        truth = pd.read_csv(truth_csv)
        truth.columns = truth.columns.str.strip().str.lower()
        theta_col = next(i for i in truth.columns if 'theta' in i)
        truth = truth[['drone','seconds_after_takeoff',theta_col,'wind_speed','wind_dir']]
        truth.rename(columns={theta_col:'theta_obs'}, inplace=True)

        # Set holders for later
        self.truth_df = truth
        self.runs_dir = runs_dir
        self.sigma = sigma_nll
        self.winsor_pct = winsor_pct
        self.X = None
        self.y = None
        self.groups = None
        self.model = None


    def assemble(self, prefix):
        """
    
        """
        # Building the features and targets for the emulator
        index = self.priors_df.index.astype(int).to_numpy()
        featurs, nlls = [], []
        for i in index:
            path_to_runs = os.path.join(self.runs_dir, f"{prefix}_{i}_output.csv")
            if not os.path.isfile(path_to_runs): continue
            columns_needed = pd.read_csv(path_to_runs, usecols=['drone','seconds_after_takeoff','theta'])
            columns_needed.rename(columns={'theta':'theta_mod'}, inplace=True)
            merge = self.truth_df.merge(columns_needed, on=['drone','seconds_after_takeoff']).dropna()
            if merge.empty: continue
            nlls.append(compute_nll(merge['theta_mod'], merge['theta_obs'], self.sigma))
            wind_radians = merge['wind_dir']
            featurs.append([merge['wind_speed'].mean(), np.sin(wind_radians).mean(), np.cos(wind_radians).mean()])
        X_prior = self.priors_df.loc[index].values
        X_feature = np.vstack(featurs)
        self.X = np.hstack([X_prior, X_feature])
        y_nll = np.array(nlls)
        if self.winsor_pct < 100:
            cap = np.percentile(y_nll, self.winsor_pct)
            self.y = np.clip(y_nll, None, cap)
        else:
            self.y = y_nll
        self.groups = index


    def build_emulator(self, restarts=10, random_state=100699):
        kernel = (
            ConstantKernel(1.0,(1e-2,1e2)) * (
                RBF(20,(1e1,1e2)) + Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-1,1e1)) +
                RationalQuadratic(length_scale=15, alpha=0.1, length_scale_bounds=(1,1e2), alpha_bounds='fixed')
            ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6,1e2))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=False,
            n_restarts_optimizer=restarts, random_state=random_state
        )
        pipe_regress = Pipeline([('scale_X', MinMaxScaler()), ('gpr', gpr)])
        pipe_transf = Pipeline([
            ('sqrt', FunctionTransformer(np.sqrt, inverse_func=lambda x: x**2, validate=True)),
            ('scale_y', MinMaxScaler())
        ])
        self.model = TransformedTargetRegressor(regressor=pipe_regress, transformer=pipe_transf)
        self.model.fit(self.X, self.y)


    def predict(self, X):
        return self.model.predict(X)


    def predict_std(self, X):
        return self.model.regressor_.named_steps['gpr'].predict(
            self.model.regressor_.named_steps['scale_X'].transform(X), return_std=True)[1]


    def run_RAM(self, bounds, n_chains=1, n_steps=200000, burn_in=10000,
                step_size=0.0001, beta=1.5, random_state=100699):
        chain = []
        for i in range(n_chains):
            seed = random_state 
            rng = np.random.RandomState(random_state)
            init = [rng.uniform(l,h) for l,h in bounds.values()]
            df = self.single_RAM(bounds, n_steps, burn_in, step_size, beta, init, seed)
            df['chain'] = i; chain.append(df)
        return pd.concat(chain, ignore_index=True)


    def single_RAM(self, bounds, n_steps, burn_in, step_size, beta, init_theta, random_state = 100699):
        params_keys = list(bounds.keys())
        params_values = np.array(list(bounds.values()))
        dimentions = len(params_keys)
        rando = np.random.RandomState(random_state)
        theta = np.array(init_theta)
        wind = self.X[:, -3:].mean(axis=0)
        chain = np.zeros((n_steps, dimentions)); acc = 0

        # initial 
        X_initial = np.hstack([theta, wind]).reshape(1, -1)
        m_initial = self.predict(X_initial)[0]
        s_initial = self.predict_std(X_initial)[0]
        L_initial = m_initial + beta * s_initial

        sigma = step_size
        target = 0.234
        adapt = 0.6

        for t in trange(n_steps, desc='RAM'):
            prop = theta + rando.normal(scale=sigma, size=dimentions)
            if np.any(prop < params_values[:, 0]) or np.any(prop > params_values[:, 1]):
                chain[t] = theta
            else:
                X_prior = np.hstack([prop, wind]).reshape(1, -1)
                m_prior = self.predict(X_prior)[0]; sp = self.predict_std(X_prior)[0]
                L_prior = m_prior + beta * sp
                alpha = np.exp(-(L_prior - L_initial))
                ok = rando.rand() < alpha
                if ok: theta, L_initial = prop, L_prior
                acc += int(ok)
                chain[t] = theta
                if t < burn_in:
                    gamma = (t+1)**(-adapt); sigma *= np.exp(gamma*(ok - target))
        print(f"Chain done")
        df = pd.DataFrame(chain, columns=params_keys)
        return df.iloc[burn_in:].reset_index(drop=True)





def main():
    random_state=100699

    PRIOR_SYNTH = '../Data/priors/priors_synthetic_truth2.csv'
    TRUTH_SYNTH = '../Data/Truth_files/zac_shf_truth2_output_perturbed.csv'
    RUNS_DIR_SYNTH = '../Data/Processed_files/Processed_files_truth2_calibrated'
    PREFIX_SYNTH = 'zac_shf_calibrated_truth2'

    print('Synthetic case:')
    emulator = EmulatorSampler(PRIOR_SYNTH, TRUTH_SYNTH, RUNS_DIR_SYNTH)
    print('Assembling data...')
    emulator.assemble(PREFIX_SYNTH)
    emulator.build_emulator(random_state=random_state)
    print('Emulator 80/20 train-test evaluation:')
    X_tr, X_te, y_tr, y_te = train_test_split(emulator.X, emulator.y, test_size=0.2, random_state=random_state)
    emulator.model.fit(X_tr, y_tr)
    y_pred_tr = emulator.predict(X_tr)
    y_pred_te = emulator.predict(X_te)
    print(f"Train R2={r2_score(y_tr, y_pred_tr):.3f}, RMSE={mean_squared_error(y_tr, y_pred_tr, squared=False):.3f}")
    print(f"Test  R2={r2_score(y_te, y_pred_te):.3f}, RMSE={mean_squared_error(y_te, y_pred_te, squared=False):.3f}")

    # PLotting
    mn = min(y_tr.min(), y_pred_tr.min(), y_te.min(), y_pred_te.min())
    mx = max(y_tr.max(), y_pred_tr.max(), y_te.max(), y_pred_te.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    ax1 = axes[0]
    ax1.scatter(y_tr, y_pred_tr, color=tab10[0], alpha=0.9)
    ax1.plot([mn, mx], [mn, mx], 'k--', label='1:1')
    ax1.set_xlabel('True NLL')
    ax1.set_ylabel('Predicted NLL')
    ax1.set_title('Train Synthetic')
    ax1.legend()
    # Test parity
    ax2 = axes[1]
    ax2.scatter(y_te, y_pred_te, color=tab10[1], alpha=0.9)
    ax2.plot([mn, mx], [mn, mx], 'k--', label='1:1')
    ax2.set_xlabel('True NLL')
    ax2.set_title('Test Synthetic')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('../Tables_and_Figures/Em_TestTrain_Synth.png', format='png', dpi=500, bbox_inches='tight')
    plt.show()

    df_p = pd.read_csv(PRIOR_SYNTH, index_col=0)
    bounds = {i: (df_p[i].min(), df_p[i].max()) for i in df_p.columns}

    print(f"Running RAM MCMC:")
    n_chains = 1
    n_steps = 200000
    burn_in = 10000
    beta = 1.5
    step_size = 0.0001
    samples = emulator.run_RAM(bounds, n_chains=n_chains, n_steps=n_steps, burn_in=burn_in, step_size= step_size, beta = beta, random_state=random_state)
    per_chain = n_steps - burn_in
    total = per_chain * n_chains

    print(f"Removed{burn_in} burn-in samples per chain: {per_chain} samples retained per chain, {total} total.")
    samples.to_csv('../Data/Posteriors/MCMC_Chains_Synth.csv', index=False)
    print(samples.describe())


    #Flight 21 case:
    PRIOR_FLIGHT21 = '../Data/priors/priors_flight_21.csv'
    TRUTH_FLIGHT21 = '../Data/Truth_files/Processed_flight_21.csv'
    RUNS_DIR_FLIGHT21 = '../Data/Processed_files/Processed_files_flight_21_calibrated'
    PREFIX_FLIGHT21 = 'zac_shf_calibrated_flight21'

    print('Flight 21 case:')
    emulator = EmulatorSampler(PRIOR_FLIGHT21, TRUTH_FLIGHT21, RUNS_DIR_FLIGHT21)
    print('Assembling data...')
    emulator.assemble(PREFIX_FLIGHT21)
    emulator.build_emulator()
    print('Emulator 80/20 train-test evaluation:')
    X_tr, X_te, y_tr, y_te = train_test_split(emulator.X, emulator.y, test_size=0.2, random_state=random_state)
    emulator.model.fit(X_tr, y_tr)
    y_pred_tr = emulator.predict(X_tr)
    y_pred_te = emulator.predict(X_te)
    print(f"Train R2={r2_score(y_tr, y_pred_tr):.3f}, RMSE={mean_squared_error(y_tr, y_pred_tr, squared=False):.3f}")
    print(f"Test  R2={r2_score(y_te, y_pred_te):.3f}, RMSE={mean_squared_error(y_te, y_pred_te, squared=False):.3f}")

    # PLotting
    mn = min(y_tr.min(), y_pred_tr.min(), y_te.min(), y_pred_te.min())
    mx = max(y_tr.max(), y_pred_tr.max(), y_te.max(), y_pred_te.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    ax1 = axes[0]
    ax1.scatter(y_tr, y_pred_tr, color=tab10[0], alpha=0.9)
    ax1.plot([mn, mx], [mn, mx], 'k--', label='1:1')
    ax1.set_xlabel('True NLL')
    ax1.set_ylabel('Predicted NLL')
    ax1.set_title('Train Synthetic')
    ax1.legend()
    # Test parity
    ax2 = axes[1]
    ax2.scatter(y_te, y_pred_te, color=tab10[1], alpha=0.9)
    ax2.plot([mn, mx], [mn, mx], 'k--', label='1:1')
    ax2.set_xlabel('True NLL')
    ax2.set_title('Test Synthetic')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('../Tables_and_Figures/Em_TestTrain_Flight21.png', format='png', dpi=500, bbox_inches='tight')
    plt.show()


    df_p = pd.read_csv(PRIOR_FLIGHT21, index_col=0)
    bounds = {i: (df_p[i].min(), df_p[i].max()) for i in df_p.columns}

    print(f"Running RAM MCMC:")
    n_chains = 1
    n_steps = 200000
    burn_in = 10000
    samples = emulator.run_RAM(bounds, n_chains=n_chains, n_steps=n_steps, burn_in=burn_in,random_state=random_state)
    per_chain = n_steps - burn_in
    total = per_chain * n_chains
    print(f"Removed{burn_in} burn-in samples per chain: {per_chain} samples retained per chain, {total} total.")
    samples.to_csv('../Data/Posteriors/MCMC_Chains_Flight21.csv', index=False)
    print(samples.describe())


if __name__ == '__main__':
    main()






































