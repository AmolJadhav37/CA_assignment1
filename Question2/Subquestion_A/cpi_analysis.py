#!/usr/bin/env python3
"""
CPI Stack Regression Analysis for Computer Architecture Assignment
Analyzes performance counter data and builds linear regression models for CPI prediction
Updated to handle the specific perf output format from the provided data files
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import re
import warnings
import os
warnings.filterwarnings('ignore')

class CPIStackAnalyzer:
    def __init__(self, file_path, benchmark_name):
        self.file_path = file_path
        self.benchmark_name = benchmark_name
        self.data = None
        self.regression_data = None
        self.model = None
        self.features = None
        self.target = None
        
    def parse_perf_data(self):
        """Parse perf stat output file and extract performance counter data"""
        print(f"\n=== Parsing {self.benchmark_name} data from {self.file_path} ===")
        
        data_rows = []
        
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
                
            # Split by lines and process each line
            lines = content.strip().split('\n')
            current_row = {}
            
            for line in lines:
                line = line.strip()
                if not line or 'beta-' in line or line.startswith('#'):
                    continue
                
                # Parse the perf stat format: time count event_name
                # Format: timestamp count event_name [additional info]
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        time_val = float(parts[0])
                        count_str = parts[1]
                        event_name = parts[2]
                        
                        # Handle <not supported> and <not counted>
                        if '<not' in count_str:
                            count_val = 0
                        else:
                            # Remove commas from count
                            count_val = float(count_str.replace(',', ''))
                        
                        # Initialize new row if time changed or first entry
                        if 'time' not in current_row or current_row['time'] != time_val:
                            if current_row:  # Save previous row if exists
                                data_rows.append(current_row.copy())
                            current_row = {'time': time_val}
                        
                        # Add the counter value
                        current_row[event_name] = count_val
                        
                    except (ValueError, IndexError) as e:
                        continue
            
            # Add the last row
            if current_row:
                data_rows.append(current_row)
        
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found!")
            return None
        
        if not data_rows:
            print("No valid data found in file!")
            return None
            
        # Convert to DataFrame
        self.data = pd.DataFrame(data_rows)
        
        # Fill NaN values with 0 for missing counters
        self.data = self.data.fillna(0)
        
        print(f"Successfully parsed {len(self.data)} data points")
        print(f"Available counters: {[col for col in self.data.columns if col != 'time']}")
        
        return self.data
    
    def prepare_regression_data(self):
        """Prepare data for regression analysis"""
        if self.data is None:
            print("No data available. Please parse data first.")
            return None
        
        print(f"\n=== Preparing regression data for {self.benchmark_name} ===")
        
        # Calculate CPI (target variable)
        if 'cycles' in self.data.columns and 'instructions' in self.data.columns:
            # Filter out zero instruction intervals
            valid_intervals = (self.data['instructions'] > 0) & (self.data['cycles'] > 0)
            self.regression_data = self.data[valid_intervals].copy()
            self.regression_data['CPI'] = self.regression_data['cycles'] / self.regression_data['instructions']
        else:
            print("Error: cycles and instructions counters not found!")
            return None
        
        # Define miss rate features for CPI stack
        features_to_create = []
        
        # L1 Data Cache miss rate
        if 'L1-dcache-loads' in self.regression_data.columns and 'L1-dcache-load-misses' in self.regression_data.columns:
            self.regression_data['L1_dcache_miss_rate'] = (
                self.regression_data['L1-dcache-load-misses'] / 
                np.maximum(self.regression_data['L1-dcache-loads'], 1)
            )
            features_to_create.append('L1_dcache_miss_rate')
        
        # L1 Instruction Cache miss rate
        if 'L1-icache-load-misses' in self.regression_data.columns:
            # Use instructions as proxy for icache loads if not available
            icache_loads = self.regression_data.get('L1-icache-loads', self.regression_data['instructions'])
            self.regression_data['L1_icache_miss_rate'] = (
                self.regression_data['L1-icache-load-misses'] / 
                np.maximum(icache_loads, 1)
            )
            features_to_create.append('L1_icache_miss_rate')
        
        # LLC (Last Level Cache) miss rate
        if 'LLC-loads' in self.regression_data.columns and 'LLC-load-misses' in self.regression_data.columns:
            self.regression_data['LLC_miss_rate'] = (
                self.regression_data['LLC-load-misses'] / 
                np.maximum(self.regression_data['LLC-loads'], 1)
            )
            features_to_create.append('LLC_miss_rate')
        
        # Overall cache miss rate if specific LLC not available
        if 'cache-references' in self.regression_data.columns and 'cache-misses' in self.regression_data.columns:
            self.regression_data['cache_miss_rate'] = (
                self.regression_data['cache-misses'] / 
                np.maximum(self.regression_data['cache-references'], 1)
            )
            features_to_create.append('cache_miss_rate')
        
        # Data TLB miss rate
        if 'dTLB-loads' in self.regression_data.columns and 'dTLB-load-misses' in self.regression_data.columns:
            self.regression_data['dTLB_miss_rate'] = (
                self.regression_data['dTLB-load-misses'] / 
                np.maximum(self.regression_data['dTLB-loads'], 1)
            )
            features_to_create.append('dTLB_miss_rate')
        
        # Instruction TLB miss rate
        if 'iTLB-load-misses' in self.regression_data.columns:
            # Use instructions as proxy for iTLB loads if not available
            itlb_loads = self.regression_data.get('iTLB-loads', self.regression_data['instructions'])
            self.regression_data['iTLB_miss_rate'] = (
                self.regression_data['iTLB-load-misses'] / 
                np.maximum(itlb_loads, 1)
            )
            features_to_create.append('iTLB_miss_rate')
        
        # Branch misprediction rate
        if 'branches' in self.regression_data.columns and 'branch-misses' in self.regression_data.columns:
            self.regression_data['branch_miss_rate'] = (
                self.regression_data['branch-misses'] / 
                np.maximum(self.regression_data['branches'], 1)
            )
            features_to_create.append('branch_miss_rate')
        
        # Store TLB miss rate if available
        if 'dTLB-stores' in self.regression_data.columns and 'dTLB-store-misses' in self.regression_data.columns:
            self.regression_data['dTLB_store_miss_rate'] = (
                self.regression_data['dTLB-store-misses'] / 
                np.maximum(self.regression_data['dTLB-stores'], 1)
            )
            features_to_create.append('dTLB_store_miss_rate')
        
        # Set features and target
        self.features = [f for f in features_to_create if f in self.regression_data.columns]
        self.target = 'CPI'
        
        # Remove infinite and NaN values
        self.regression_data = self.regression_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for feature in self.features + [self.target]:
            mean_val = self.regression_data[feature].mean()
            std_val = self.regression_data[feature].std()
            self.regression_data = self.regression_data[
                abs(self.regression_data[feature] - mean_val) <= 3 * std_val
            ]
        
        print(f"Features for regression: {self.features}")
        print(f"Data points after cleaning: {len(self.regression_data)}")
        print(f"CPI range: {self.regression_data['CPI'].min():.4f} - {self.regression_data['CPI'].max():.4f}")
        
        return self.regression_data
    
    def build_regression_model(self):
        """Build linear regression model with non-negative constraints"""
        if self.regression_data is None or not self.features:
            print("No regression data available. Please prepare data first.")
            return None
        
        print(f"\n=== Building regression model for {self.benchmark_name} ===")
        
        X = self.regression_data[self.features].values
        y = self.regression_data[self.target].values
        
        # Use LinearRegression with positive constraint approximation
        self.model = LinearRegression(positive=True)
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate adjusted R²
        n = len(y)
        p = len(self.features)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if p > 0 else r2
        
        # Calculate F-statistic and p-value
        from scipy import stats
        if r2 < 1.0 and p > 0:
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        else:
            f_stat = np.inf if r2 == 1.0 else 0
            p_value = 0.0 if r2 == 1.0 else 1.0
        
        # Store results
        self.results = {
            'coefficients': dict(zip(self.features, self.model.coef_)),
            'intercept': self.model.intercept_,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'f_statistic': f_stat,
            'p_value': p_value,
            'n_samples': n,
            'n_features': p,
            'y_true': y,
            'y_pred': y_pred,
            'residuals': y - y_pred
        }
        
        print(f"Model built successfully with {n} samples and {p} features")
        return self.model
    
    def print_results(self):
        """Print comprehensive model results"""
        if not hasattr(self, 'results'):
            print("No model results available.")
            return
        
        print(f"\n" + "="*60)
        print(f"CPI STACK REGRESSION RESULTS FOR {self.benchmark_name.upper()}")
        print(f"="*60)
        
        print(f"\nModel Equation:")
        equation = f"CPI = {self.results['intercept']:.6f}"
        for feature, coef in self.results['coefficients'].items():
            equation += f" + {coef:.6f} * {feature}"
        print(equation)
        
        print(f"\nCoefficients (CPI Stack Components):")
        print(f"{'Component':<30} {'Coefficient':<15} {'Interpretation'}")
        print("-" * 70)
        print(f"{'Base CPI (Intercept)':<30} {self.results['intercept']:<15.6f} {'Perfect execution'}")
        
        for feature, coef in self.results['coefficients'].items():
            interpretation = self.get_coefficient_interpretation(feature, coef)
            print(f"{feature:<30} {coef:<15.6f} {interpretation}")
        
        print(f"\nModel Quality Metrics:")
        print(f"RMSE:                     {self.results['rmse']:.6f}")
        print(f"MAE:                      {self.results['mae']:.6f}")
        print(f"R²:                       {self.results['r2']:.6f}")
        print(f"Adjusted R²:              {self.results['adj_r2']:.6f}")
        print(f"F-statistic:              {self.results['f_statistic']:.2f}")
        print(f"p-value:                  {self.results['p_value']:.2e}")
        print(f"Number of samples:        {self.results['n_samples']}")
        print(f"Number of features:       {self.results['n_features']}")
        
        # Model quality interpretation
        print(f"\nModel Quality Interpretation:")
        if self.results['r2'] > 0.8:
            print("- Excellent model fit (R² > 0.8)")
        elif self.results['r2'] > 0.6:
            print("- Good model fit (R² > 0.6)")
        elif self.results['r2'] > 0.4:
            print("- Moderate model fit (R² > 0.4)")
        else:
            print("- Poor model fit (R² < 0.4)")
            
        if self.results['p_value'] < 0.01:
            print("- Model is highly statistically significant (p < 0.01)")
        elif self.results['p_value'] < 0.05:
            print("- Model is statistically significant (p < 0.05)")
        else:
            print("- Model may not be statistically significant (p >= 0.05)")
    
    def get_coefficient_interpretation(self, feature, coef):
        """Get human-readable interpretation of coefficient"""
        interpretations = {
            'L1_dcache_miss_rate': 'L1 data cache penalty',
            'L1_icache_miss_rate': 'L1 instruction cache penalty',
            'LLC_miss_rate': 'Last level cache penalty',
            'cache_miss_rate': 'General cache penalty',
            'dTLB_miss_rate': 'Data TLB penalty',
            'iTLB_miss_rate': 'Instruction TLB penalty',
            'branch_miss_rate': 'Branch misprediction penalty',
            'dTLB_store_miss_rate': 'Store TLB penalty'
        }
        return interpretations.get(feature, 'Performance penalty')
    
    def analyze_phases(self, n_phases=3):
        """Analyze different execution phases"""
        if self.regression_data is None:
            print("No regression data available.")
            return None
        
        print(f"\n=== Phase Analysis for {self.benchmark_name} ===")
        
        # Divide data into phases
        phase_size = len(self.regression_data) // n_phases
        phases = {}
        
        for i in range(n_phases):
            start_idx = i * phase_size
            if i == n_phases - 1:  # Last phase gets remaining data
                end_idx = len(self.regression_data)
            else:
                end_idx = (i + 1) * phase_size
            
            phase_data = self.regression_data.iloc[start_idx:end_idx]
            phase_name = f"Phase_{i+1}"
            phases[phase_name] = phase_data
            
            print(f"\n{phase_name} (samples {start_idx}-{end_idx-1}):")
            print(f"  Mean CPI:     {phase_data['CPI'].mean():.4f} ± {phase_data['CPI'].std():.4f}")
            
            # Show mean miss rates for this phase
            for feature in self.features:
                if feature in phase_data.columns:
                    mean_val = phase_data[feature].mean()
                    std_val = phase_data[feature].std()
                    print(f"  Mean {feature:<20}: {mean_val:.6f} ± {std_val:.6f}")
        
        return phases
    
    def create_visualizations(self):
        """Create visualization plots"""
        if not hasattr(self, 'results'):
            print("No model results available for visualization.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'CPI Stack Analysis - {self.benchmark_name}', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Actual vs Predicted CPI
        axes[0, 0].scatter(self.results['y_true'], self.results['y_pred'], alpha=0.6, s=30)
        axes[0, 0].plot([self.results['y_true'].min(), self.results['y_true'].max()], 
                        [self.results['y_true'].min(), self.results['y_true'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual CPI', fontsize=12)
        axes[0, 0].set_ylabel('Predicted CPI', fontsize=12)
        axes[0, 0].set_title(f'Actual vs Predicted CPI\n(R² = {self.results["r2"]:.3f})', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        axes[0, 1].scatter(self.results['y_pred'], self.results['residuals'], alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted CPI', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance (coefficients) - FIXED LABELS
        coef_names = list(self.results['coefficients'].keys())
        coef_values = list(self.results['coefficients'].values())
        
        # Add intercept
        coef_names.append('Intercept')
        coef_values.append(self.results['intercept'])
        
        # Shorten feature names for better display
        display_names = []
        for name in coef_names:
            if name == 'Intercept':
                display_names.append('Base CPI')
            elif 'L1_dcache' in name:
                display_names.append('L1 D-Cache')
            elif 'L1_icache' in name:
                display_names.append('L1 I-Cache')
            elif 'LLC_miss' in name:
                display_names.append('LLC Miss')
            elif 'cache_miss' in name:
                display_names.append('Cache Miss')
            elif 'dTLB_miss' in name:
                display_names.append('Data TLB')
            elif 'iTLB_miss' in name:
                display_names.append('Instr TLB')
            elif 'branch_miss' in name:
                display_names.append('Branch Miss')
            elif 'dTLB_store' in name:
                display_names.append('Store TLB')
            else:
                display_names.append(name[:15])  # Truncate long names
        
        colors = ['skyblue' if x >= 0 else 'lightcoral' for x in coef_values]
        bars = axes[1, 0].barh(display_names, coef_values, color=colors)
        axes[1, 0].set_xlabel('Coefficient Value', fontsize=12)
        axes[1, 0].set_title('CPI Stack Components', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars with better positioning
        for bar, value in zip(bars, coef_values):
            width = bar.get_width()
            label_x = width + (max(coef_values) * 0.02) if width >= 0 else width - (max(coef_values) * 0.02)
            axes[1, 0].text(label_x, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # Adjust margins for coefficient plot
        axes[1, 0].margins(x=0.15)
        
        # 4. CPI over time
        time_data = self.regression_data['time'].values if 'time' in self.regression_data.columns else range(len(self.regression_data))
        axes[1, 1].plot(time_data, self.regression_data['CPI'], alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('Time (seconds)' if 'time' in self.regression_data.columns else 'Sample Index', fontsize=12)
        axes[1, 1].set_ylabel('CPI', fontsize=12)
        axes[1, 1].set_title('CPI Over Time', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Improve spacing between subplots
        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.25, wspace=0.25)
        
        # Save the plot
        filename = f'{self.benchmark_name.lower()}_cpi_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        
        return filename


def main():
    """Main analysis function"""
    print("CPI Stack Regression Analysis Tool")
    print("="*50)
    
    # File paths - adjust these to match your actual file locations
    streamcluster_file = os.path.expanduser("~/assignments/CA_assignment1/Question2/Subquestion_A/parsec-benchmark/streamcluster_detailed_data.txt")
    gap_file = os.path.expanduser("~/assignments/CA_assignment1/Question2/Subquestion_A/gapbs/gap_bfs_data.txt")
    
    results_summary = []
    
    # Analyze Streamcluster
    print("\n" + "="*60)
    print("ANALYZING STREAMCLUSTER BENCHMARK")
    print("="*60)
    
    streamcluster_analyzer = CPIStackAnalyzer(streamcluster_file, "Streamcluster")
    if streamcluster_analyzer.parse_perf_data() is not None:
        streamcluster_analyzer.prepare_regression_data()
        streamcluster_analyzer.build_regression_model()
        streamcluster_analyzer.print_results()
        streamcluster_phases = streamcluster_analyzer.analyze_phases()
        streamcluster_plot = streamcluster_analyzer.create_visualizations()
        
        if hasattr(streamcluster_analyzer, 'results'):
            results_summary.append({
                'Benchmark': 'Streamcluster',
                'R²': streamcluster_analyzer.results['r2'],
                'RMSE': streamcluster_analyzer.results['rmse'],
                'Adj_R²': streamcluster_analyzer.results['adj_r2'],
                'Mean_CPI': streamcluster_analyzer.regression_data['CPI'].mean(),
                'Features': len(streamcluster_analyzer.features)
            })
    
    # Analyze GAP BFS
    print("\n" + "="*60)
    print("ANALYZING GAP BFS BENCHMARK")
    print("="*60)
    
    gap_analyzer = CPIStackAnalyzer(gap_file, "GAP_BFS")
    if gap_analyzer.parse_perf_data() is not None:
        gap_analyzer.prepare_regression_data()
        gap_analyzer.build_regression_model()
        gap_analyzer.print_results()
        gap_phases = gap_analyzer.analyze_phases()
        gap_plot = gap_analyzer.create_visualizations()
        
        if hasattr(gap_analyzer, 'results'):
            results_summary.append({
                'Benchmark': 'GAP_BFS',
                'R²': gap_analyzer.results['r2'],
                'RMSE': gap_analyzer.results['rmse'],
                'Adj_R²': gap_analyzer.results['adj_r2'],
                'Mean_CPI': gap_analyzer.regression_data['CPI'].mean(),
                'Features': len(gap_analyzer.features)
            })
    
    # Comparison Summary
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*60)
    
    if len(results_summary) > 0:
        comparison_df = pd.DataFrame(results_summary)
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        print(f"\nKey Observations:")
        if len(results_summary) == 2:
            sc_cpi = results_summary[0]['Mean_CPI']
            gap_cpi = results_summary[1]['Mean_CPI']
            print(f"- Streamcluster Mean CPI: {sc_cpi:.4f}")
            print(f"- GAP BFS Mean CPI: {gap_cpi:.4f}")
            
            if sc_cpi > gap_cpi:
                print(f"- Streamcluster has {((sc_cpi/gap_cpi-1)*100):.1f}% higher CPI than GAP BFS")
            else:
                print(f"- GAP BFS has {((gap_cpi/sc_cpi-1)*100):.1f}% higher CPI than Streamcluster")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    
    # Check for generated files
    generated_files = []
    for filename in ['streamcluster_cpi_analysis.png', 'gap_bfs_cpi_analysis.png']:
        if os.path.exists(filename):
            generated_files.append(filename)
            print(f"✓ {filename}")
    
    if generated_files:
        print(f"\nYou can view these plots to analyze:")
        print("1. CPI prediction accuracy")
        print("2. Model residuals")
        print("3. CPI stack components (coefficients)")
        print("4. CPI variation over time")
    
    print("\nFor your report, focus on:")
    print("- Compare R² values between benchmarks")
    print("- Analyze which miss types have highest coefficients")
    print("- Discuss phase differences in CPI behavior")
    print("- Compare overall CPI characteristics between workloads")

if __name__ == "__main__":
    main()