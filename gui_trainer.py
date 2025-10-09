"""
6 qui Prend - RL Training GUI
Graphical interface for training and comparing RL agents
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import subprocess
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class RLTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("6 qui Prend - RL Training Suite")
        self.root.geometry("1200x800")
        
        # Project paths
        self.project_dir = Path(__file__).parent
        self.scripts_dir = self.project_dir / 'scripts'
        self.models_dir = self.project_dir / 'saved_models'
        self.results_dir = self.project_dir / 'results'
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Training process tracking
        self.current_process = None
        self.training_active = False
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_ppo_tab()
        self.create_reinforce_tab()
        self.create_imitation_tab()
        self.create_selfplay_tab()
        self.create_compare_tab()
        self.create_results_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_ppo_tab(self):
        """Create PPO training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="PPO Training")
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(tab, text="Training Parameters", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        row = 0
        
        # Episodes
        ttk.Label(left_frame, text="Episodes:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_episodes = ttk.Scale(left_frame, from_=100, to=5000, orient=tk.HORIZONTAL, length=200)
        self.ppo_episodes.set(3000)
        self.ppo_episodes.grid(row=row, column=1, pady=5)
        self.ppo_episodes_label = ttk.Label(left_frame, text="3000")
        self.ppo_episodes_label.grid(row=row, column=2, padx=5)
        self.ppo_episodes.configure(command=lambda v: self.ppo_episodes_label.config(text=f"{int(float(v))}"))
        row += 1
        
        # Opponent
        ttk.Label(left_frame, text="Opponent:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_opponent = ttk.Combobox(left_frame, values=['random', 'greedy'], state='readonly', width=20)
        self.ppo_opponent.set('greedy')
        self.ppo_opponent.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Learning rate
        ttk.Label(left_frame, text="Learning Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_lr = ttk.Entry(left_frame, width=22)
        self.ppo_lr.insert(0, "0.0003")
        self.ppo_lr.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Gamma
        ttk.Label(left_frame, text="Gamma (discount):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_gamma = ttk.Entry(left_frame, width=22)
        self.ppo_gamma.insert(0, "0.99")
        self.ppo_gamma.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # GAE Lambda
        ttk.Label(left_frame, text="GAE Lambda:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_gae = ttk.Entry(left_frame, width=22)
        self.ppo_gae.insert(0, "0.95")
        self.ppo_gae.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Update epochs
        ttk.Label(left_frame, text="Update Epochs:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_epochs = ttk.Spinbox(left_frame, from_=1, to=10, width=20)
        self.ppo_epochs.set(4)
        self.ppo_epochs.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Clip epsilon
        ttk.Label(left_frame, text="Clip Epsilon:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_clip = ttk.Entry(left_frame, width=22)
        self.ppo_clip.insert(0, "0.2")
        self.ppo_clip.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Entropy coefficient
        ttk.Label(left_frame, text="Entropy Coef:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_entropy = ttk.Entry(left_frame, width=22)
        self.ppo_entropy.insert(0, "0.01")
        self.ppo_entropy.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Save path
        ttk.Label(left_frame, text="Save Model As:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ppo_save = ttk.Entry(left_frame, width=22)
        self.ppo_save.insert(0, "ppo_agent.pth")
        self.ppo_save.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Pretrained model (optional)
        ttk.Label(left_frame, text="Load Pretrained:").grid(row=row, column=0, sticky=tk.W, pady=5)
        pretrained_frame = ttk.Frame(left_frame)
        pretrained_frame.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.ppo_pretrained = ttk.Entry(pretrained_frame, width=15)
        self.ppo_pretrained.pack(side=tk.LEFT)
        ttk.Button(pretrained_frame, text="Browse", 
                  command=lambda: self.browse_model(self.ppo_pretrained)).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        self.ppo_train_btn = ttk.Button(button_frame, text="Start Training", 
                                        command=self.train_ppo, style='Accent.TButton')
        self.ppo_train_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Output log
        right_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ppo_log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                                 height=30, width=60, font=('Courier', 9))
        self.ppo_log.pack(fill=tk.BOTH, expand=True)
        
    def create_reinforce_tab(self):
        """Create REINFORCE training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="REINFORCE")
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(tab, text="Training Parameters", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        row = 0
        
        # Episodes
        ttk.Label(left_frame, text="Episodes:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rf_episodes = ttk.Scale(left_frame, from_=100, to=5000, orient=tk.HORIZONTAL, length=200)
        self.rf_episodes.set(1000)
        self.rf_episodes.grid(row=row, column=1, pady=5)
        self.rf_episodes_label = ttk.Label(left_frame, text="1000")
        self.rf_episodes_label.grid(row=row, column=2, padx=5)
        self.rf_episodes.configure(command=lambda v: self.rf_episodes_label.config(text=f"{int(float(v))}"))
        row += 1
        
        # Opponent
        ttk.Label(left_frame, text="Opponent:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rf_opponent = ttk.Combobox(left_frame, values=['random', 'greedy'], state='readonly', width=20)
        self.rf_opponent.set('random')
        self.rf_opponent.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Learning rate
        ttk.Label(left_frame, text="Learning Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rf_lr = ttk.Entry(left_frame, width=22)
        self.rf_lr.insert(0, "0.001")
        self.rf_lr.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Gamma
        ttk.Label(left_frame, text="Gamma (discount):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rf_gamma = ttk.Entry(left_frame, width=22)
        self.rf_gamma.insert(0, "0.99")
        self.rf_gamma.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Save path
        ttk.Label(left_frame, text="Save Model As:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rf_save = ttk.Entry(left_frame, width=22)
        self.rf_save.insert(0, "reinforce_agent.pth")
        self.rf_save.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Start Training", 
                  command=self.train_reinforce, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Output log
        right_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.rf_log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                               height=30, width=60, font=('Courier', 9))
        self.rf_log.pack(fill=tk.BOTH, expand=True)
        
    def create_imitation_tab(self):
        """Create imitation learning tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Imitation Learning")
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(tab, text="Pre-training Parameters", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        row = 0
        
        # Episodes
        ttk.Label(left_frame, text="Episodes:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.im_episodes = ttk.Scale(left_frame, from_=100, to=3000, orient=tk.HORIZONTAL, length=200)
        self.im_episodes.set(1000)
        self.im_episodes.grid(row=row, column=1, pady=5)
        self.im_episodes_label = ttk.Label(left_frame, text="1000")
        self.im_episodes_label.grid(row=row, column=2, padx=5)
        self.im_episodes.configure(command=lambda v: self.im_episodes_label.config(text=f"{int(float(v))}"))
        row += 1
        
        # Learning rate
        ttk.Label(left_frame, text="Learning Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.im_lr = ttk.Entry(left_frame, width=22)
        self.im_lr.insert(0, "0.001")
        self.im_lr.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Save path
        ttk.Label(left_frame, text="Save Model As:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.im_save = ttk.Entry(left_frame, width=22)
        self.im_save.insert(0, "pretrained_agent.pth")
        self.im_save.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Info
        info_label = ttk.Label(left_frame, text="Pre-trains agent by imitating\ngreedy agent (behavioral cloning)",
                              foreground='blue', justify=tk.LEFT)
        info_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Start Pre-training", 
                  command=self.train_imitation, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Output log
        right_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.im_log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                               height=30, width=60, font=('Courier', 9))
        self.im_log.pack(fill=tk.BOTH, expand=True)
        
    def create_selfplay_tab(self):
        """Create self-play training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Self-Play")
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(tab, text="Self-Play Parameters", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        row = 0
        
        # Generations
        ttk.Label(left_frame, text="Generations:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sp_generations = ttk.Spinbox(left_frame, from_=1, to=50, width=20)
        self.sp_generations.set(10)
        self.sp_generations.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Episodes per generation
        ttk.Label(left_frame, text="Episodes/Gen:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sp_episodes = ttk.Scale(left_frame, from_=50, to=1000, orient=tk.HORIZONTAL, length=200)
        self.sp_episodes.set(300)
        self.sp_episodes.grid(row=row, column=1, pady=5)
        self.sp_episodes_label = ttk.Label(left_frame, text="300")
        self.sp_episodes_label.grid(row=row, column=2, padx=5)
        self.sp_episodes.configure(command=lambda v: self.sp_episodes_label.config(text=f"{int(float(v))}"))
        row += 1
        
        # Learning rate
        ttk.Label(left_frame, text="Learning Rate:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sp_lr = ttk.Entry(left_frame, width=22)
        self.sp_lr.insert(0, "0.0003")
        self.sp_lr.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Pretrained model (optional)
        ttk.Label(left_frame, text="Load Pretrained:").grid(row=row, column=0, sticky=tk.W, pady=5)
        pretrained_frame = ttk.Frame(left_frame)
        pretrained_frame.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.sp_pretrained = ttk.Entry(pretrained_frame, width=15)
        self.sp_pretrained.pack(side=tk.LEFT)
        ttk.Button(pretrained_frame, text="Browse", 
                  command=lambda: self.browse_model(self.sp_pretrained)).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Save path
        ttk.Label(left_frame, text="Save Model As:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sp_save = ttk.Entry(left_frame, width=22)
        self.sp_save.insert(0, "selfplay_agent.pth")
        self.sp_save.grid(row=row, column=1, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Info
        info_label = ttk.Label(left_frame, text="Trains agent against improving\nversions of itself",
                              foreground='blue', justify=tk.LEFT)
        info_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Start Training", 
                  command=self.train_selfplay, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Output log
        right_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.sp_log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                               height=30, width=60, font=('Courier', 9))
        self.sp_log.pack(fill=tk.BOTH, expand=True)
        
    def create_compare_tab(self):
        """Create model comparison tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Compare Models")
        
        # Left panel - Model selection
        left_frame = ttk.LabelFrame(tab, text="Select Models to Compare", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Model list
        ttk.Label(left_frame, text="Selected Models:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Frame for listbox and scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.compare_model_list = tk.Listbox(list_frame, height=10, width=40, 
                                             yscrollcommand=scrollbar.set, 
                                             selectmode=tk.EXTENDED)
        self.compare_model_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.compare_model_list.yview)
        
        # Store model paths (parallel to listbox items)
        self.selected_model_paths = []
        
        # Buttons for managing model list
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Add Model", 
                  command=self.add_model_to_compare).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Remove Selected", 
                  command=self.remove_selected_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all_models).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Include baselines
        self.compare_baselines = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Include Baselines (Random/Greedy)", 
                       variable=self.compare_baselines).pack(anchor=tk.W, pady=5)
        
        # Number of games
        games_frame = ttk.Frame(left_frame)
        games_frame.pack(fill=tk.X, pady=5)
        ttk.Label(games_frame, text="Evaluation Games:").pack(side=tk.LEFT)
        self.compare_games = ttk.Spinbox(games_frame, from_=100, to=1000, width=10)
        self.compare_games.set(500)
        self.compare_games.pack(side=tk.LEFT, padx=5)
        
        # Run comparison button
        ttk.Button(left_frame, text="Run Comparison", 
                  command=self.run_comparison, style='Accent.TButton').pack(pady=10)
        ttk.Button(left_frame, text="Stop", 
                  command=self.stop_training).pack(pady=5)
        
        # Right panel - Output log
        right_frame = ttk.LabelFrame(tab, text="Comparison Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.compare_log = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                                    height=30, width=60, font=('Courier', 9))
        self.compare_log.pack(fill=tk.BOTH, expand=True)
        
    def create_results_tab(self):
        """Create results viewing tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Results")
        
        # Toolbar
        toolbar = ttk.Frame(tab)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Refresh Results", 
                  command=self.refresh_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Open Results Folder", 
                  command=lambda: os.startfile(self.results_dir) if sys.platform == 'win32' 
                  else subprocess.run(['xdg-open', str(self.results_dir)])).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Open Models Folder", 
                  command=lambda: os.startfile(self.models_dir) if sys.platform == 'win32' 
                  else subprocess.run(['xdg-open', str(self.models_dir)])).pack(side=tk.LEFT, padx=5)
        
        # Results list
        results_frame = ttk.LabelFrame(tab, text="Saved Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for results
        columns = ('File', 'Type', 'Date Modified')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Double-click to open
        self.results_tree.bind('<Double-1>', self.open_result_file)
        
        # Initial refresh
        self.refresh_results()
        
    def browse_model(self, entry_widget):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=self.models_dir,
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, os.path.basename(filename))
    
    def add_model_to_compare(self):
        """Add a model to the comparison list"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=self.models_dir,
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            # Add to list
            model_path = Path(filename)
            display_name = model_path.name
            
            # Check if already added
            if str(model_path) in self.selected_model_paths:
                messagebox.showinfo("Already Added", f"{display_name} is already in the list")
                return
            
            self.compare_model_list.insert(tk.END, display_name)
            self.selected_model_paths.append(str(model_path))
    
    def remove_selected_models(self):
        """Remove selected models from the comparison list"""
        selected_indices = self.compare_model_list.curselection()
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.compare_model_list.delete(index)
            del self.selected_model_paths[index]
    
    def clear_all_models(self):
        """Clear all models from the comparison list"""
        self.compare_model_list.delete(0, tk.END)
        self.selected_model_paths.clear()
            
    def run_command(self, cmd, log_widget, on_complete=None):
        """Run a command in a separate thread and stream output to log widget"""
        def target():
            try:
                self.training_active = True
                log_widget.insert(tk.END, f"$ {' '.join(cmd)}\n\n")
                log_widget.see(tk.END)
                
                # Start process
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=self.project_dir
                )
                
                # Stream output
                for line in self.current_process.stdout:
                    log_widget.insert(tk.END, line)
                    log_widget.see(tk.END)
                    self.root.update_idletasks()
                
                self.current_process.wait()
                
                if self.current_process.returncode == 0:
                    log_widget.insert(tk.END, "\n✓ Training completed successfully!\n")
                    self.status_var.set("Training completed")
                else:
                    log_widget.insert(tk.END, f"\n✗ Training failed with code {self.current_process.returncode}\n")
                    self.status_var.set("Training failed")
                    
            except Exception as e:
                log_widget.insert(tk.END, f"\n✗ Error: {str(e)}\n")
                self.status_var.set(f"Error: {str(e)}")
            finally:
                self.training_active = False
                self.current_process = None
                if on_complete:
                    on_complete()
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        
    def train_ppo(self):
        """Start PPO training"""
        if self.training_active:
            messagebox.showwarning("Training Active", "A training session is already running!")
            return
        
        self.ppo_log.delete(1.0, tk.END)
        self.status_var.set("Training PPO agent...")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.scripts_dir / 'train_ppo.py'),
            '--episodes', str(int(self.ppo_episodes.get())),
            '--opponent', self.ppo_opponent.get(),
            '--lr', self.ppo_lr.get(),
            '--gamma', self.ppo_gamma.get(),
            '--gae-lambda', self.ppo_gae.get(),
            '--update-epochs', str(self.ppo_epochs.get()),
            '--clip', self.ppo_clip.get(),
            '--entropy-coef', self.ppo_entropy.get(),
            '--save', str(self.models_dir / self.ppo_save.get())
        ]
        
        # Add pretrained if specified
        pretrained = self.ppo_pretrained.get().strip()
        if pretrained:
            cmd.extend(['--pretrained', str(self.models_dir / pretrained)])
        
        self.run_command(cmd, self.ppo_log)
        
    def train_reinforce(self):
        """Start REINFORCE training"""
        if self.training_active:
            messagebox.showwarning("Training Active", "A training session is already running!")
            return
        
        self.rf_log.delete(1.0, tk.END)
        self.status_var.set("Training REINFORCE agent...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / 'train_reinforce.py'),
            '--episodes', str(int(self.rf_episodes.get())),
            '--opponent', self.rf_opponent.get(),
            '--lr', self.rf_lr.get(),
            '--gamma', self.rf_gamma.get(),
            '--save', str(self.models_dir / self.rf_save.get())
        ]
        
        self.run_command(cmd, self.rf_log)
        
    def train_imitation(self):
        """Start imitation learning"""
        if self.training_active:
            messagebox.showwarning("Training Active", "A training session is already running!")
            return
        
        self.im_log.delete(1.0, tk.END)
        self.status_var.set("Pre-training agent with imitation learning...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / 'train_imitation.py'),
            '--episodes', str(int(self.im_episodes.get())),
            '--lr', self.im_lr.get(),
            '--save', str(self.models_dir / self.im_save.get())
        ]
        
        self.run_command(cmd, self.im_log)
        
    def train_selfplay(self):
        """Start self-play training"""
        if self.training_active:
            messagebox.showwarning("Training Active", "A training session is already running!")
            return
        
        self.sp_log.delete(1.0, tk.END)
        self.status_var.set("Training with self-play...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / 'train_selfplay.py'),
            '--generations', str(self.sp_generations.get()),
            '--episodes-per-gen', str(int(self.sp_episodes.get())),
            '--lr', self.sp_lr.get(),
            '--save', str(self.models_dir / self.sp_save.get())
        ]
        
        # Add pretrained if specified
        pretrained = self.sp_pretrained.get().strip()
        if pretrained:
            cmd.extend(['--pretrained', str(self.models_dir / pretrained)])
        
        self.run_command(cmd, self.sp_log)
        
    def run_comparison(self):
        """Run model comparison"""
        if self.training_active:
            messagebox.showwarning("Training Active", "A training session is already running!")
            return
        
        # Check at least one model is selected or baselines are included
        if not self.selected_model_paths and not self.compare_baselines.get():
            messagebox.showwarning("No Models", "Please add at least one model or include baselines!")
            return
        
        self.compare_log.delete(1.0, tk.END)
        self.status_var.set("Running comparison...")
        
        cmd = [sys.executable, str(self.scripts_dir / 'compare.py')]
        
        # Add all selected models using --models argument
        if self.selected_model_paths:
            cmd.append('--models')
            cmd.extend(self.selected_model_paths)
        
        # Add baselines flag
        if self.compare_baselines.get():
            cmd.append('--include-baselines')
        
        # Add number of games
        cmd.extend(['--games', str(self.compare_games.get())])
        
        # Set save path
        cmd.extend(['--save-plot', str(self.results_dir / 'comparison_results.png')])
        
        self.run_command(cmd, self.compare_log, on_complete=self.refresh_results)
        
    def stop_training(self):
        """Stop current training"""
        if self.current_process:
            self.current_process.terminate()
            self.status_var.set("Training stopped by user")
            self.training_active = False
        
    def refresh_results(self):
        """Refresh results list"""
        # Clear current items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add result files
        for file_path in sorted(self.results_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
            if file_path.is_file():
                file_type = "Plot" if file_path.suffix == '.png' else \
                           "Pickle" if file_path.suffix == '.pkl' else \
                           "Log" if file_path.suffix == '.log' else "Other"
                
                modified = file_path.stat().st_mtime
                import datetime
                date_str = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')
                
                self.results_tree.insert('', tk.END, values=(file_path.name, file_type, date_str))
        
        # Add model files
        for file_path in sorted(self.models_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True):
            modified = file_path.stat().st_mtime
            import datetime
            date_str = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')
            
            self.results_tree.insert('', tk.END, values=(f"[MODEL] {file_path.name}", "Model", date_str))
    
    def open_result_file(self, event):
        """Open selected result file"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        filename = item['values'][0]
        
        # Check if it's a model or result
        if filename.startswith('[MODEL]'):
            filename = filename.replace('[MODEL] ', '')
            filepath = self.models_dir / filename
        else:
            filepath = self.results_dir / filename
        
        # Open file based on type
        if filepath.suffix == '.png':
            # Open image
            if sys.platform == 'win32':
                os.startfile(filepath)
            else:
                subprocess.run(['xdg-open', str(filepath)])
        else:
            # Open in default editor
            if sys.platform == 'win32':
                os.startfile(filepath)
            else:
                subprocess.run(['xdg-open', str(filepath)])


def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Use clam theme for better appearance
    
    # Configure custom style for accent button
    style.configure('Accent.TButton', foreground='white', background='#007ACC', 
                   font=('Segoe UI', 10, 'bold'))
    
    app = RLTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
