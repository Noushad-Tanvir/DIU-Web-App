import os
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from datetime import datetime
import time
import logging
import re
from io import BytesIO 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================
# 0. Streamlit Page Config
# ============================
st.set_page_config(
    page_title="DIU Premium Admission Portal",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ============================
# COMPREHENSIVE CSS INTEGRATION
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Noto+Sans:wght@400;500;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

:root {
            --primary-color: #0d8bf2;
            --primary-hover: #0b76d1;
            --secondary-color: #f0f8ff;
            --background-color: #f7f9fc;
            --card-background: rgba(255, 255, 255, 0.8);
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --text-muted: #9ca3af;
            --border-color: #e5e7eb;
            --border-light: #f3f4f6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --sidebar-width: 16rem;
            --header-height: 4rem;
            --border-radius: 0.5rem;
            --border-radius-lg: 0.75rem;
            --border-radius-xl: 1rem;
        }

        /* Dark Mode Variables */
        .dark-mode {
            --primary-color: #3b82f6;
            --primary-hover: #2563eb;
            --secondary-color: #1e293b;
            --background-color: #0f172a;
            --card-background: rgba(255, 255, 255, 0.05);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --border-light: #1e293b;
        }

        /* Base Styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Manrope', 'Noto Sans', sans-serif;
            background-color: var(--background-color);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* Layout */
        .app-container {
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: var(--sidebar-width);
            background-color: white;
            border-right: 1px solid var(--border-color);
            padding: 1.5rem 1rem;
            display: flex;
            flex-direction: column;
            z-index: 50;
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out;
        }

        @media (min-width: 640px) {
            .sidebar {
                transform: translateX(0);
            }
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .logo {
            width: 2rem;
            height: 2rem;
            color: var(--primary-color);
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            flex-grow: 1;
        }

        .nav-link {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            border-radius: var(--border-radius);
            text-decoration: none;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }

        .nav-link:hover {
            background-color: var(--gray-100);
            color: var(--primary-color);
        }

        .nav-link.active {
            background-color: var(--primary-color);
            color: white;
        }

        .nav-icon {
            font-size: 1.25rem;
        }

        .nav-text {
            font-size: 0.875rem;
            font-weight: 500;
        }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .testimonial-box {
            background-color: var(--secondary-color);
            border-radius: var(--border-radius);
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }

        .testimonial-text {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .testimonial-author {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        .pref-button {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            border-radius: var(--border-radius);
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: left;
        }

        .pref-button:hover {
            background-color: var(--gray-100);
            color: var(--primary-color);
        }

        /* Main Content */
        .main-content {
            flex: 1;
            margin-left: 0;
            display: flex;
            flex-direction: column;
        }

        @media (min-width: 640px) {
            .main-content {
                margin-left: var(--sidebar-width);
            }
        }

        /* Header */
        .header {
            position: sticky;
            top: 0;
            z-index: 40;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: var(--header-height);
            padding: 0 1.5rem;
            background-color: rgba(247, 249, 252, 0.8);
            backdrop-filter: blur(4px);
            border-bottom: 1px solid var(--border-color);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .mobile-menu-button {
            display: block;
            padding: 0.5rem;
            border-radius: 0.375rem;
            background: none;
            border: none;
            cursor: pointer;
            color: var(--text-secondary);
        }

        @media (min-width: 640px) {
            .mobile-menu-button {
                display: none;
            }
        }

        .search-container {
            position: relative;
            display: none;
        }

        @media (min-width: 768px) {
            .search-container {
                display: block;
            }
        }

        .search-icon {
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
        }

        .search-input {
            width: 100%;
            min-width: 10rem;
            max-width: 16rem;
            height: 2.5rem;
            border-radius: var(--border-radius);
            background-color: white;
            border: 1px solid var(--border-color);
            padding-left: 2.5rem;
            font-size: 0.875rem;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(13, 139, 242, 0.2);
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .header-button {
            display: none;
            min-width: 5.25rem;
            height: 2.5rem;
            padding: 0 1rem;
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            font-weight: 700;
            letter-spacing: 0.025em;
            cursor: pointer;
            border: none;
            transition: all 0.3s ease;
            align-items: center;
            justify-content: center;
        }

        @media (min-width: 640px) {
            .header-button {
                display: flex;
            }
        }

        .primary-button {
            background-color: var(--primary-color);
            color: white;
        }

        .primary-button:hover {
            background-color: var(--primary-hover);
        }

        .secondary-button {
            background-color: var(--gray-200);
            color: var(--text-primary);
        }

        .secondary-button:hover {
            background-color: var(--gray-300);
        }

        .user-avatar {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            border: 2px solid var(--primary-color);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Main Area */
        .main-area {
            flex: 1;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .page-title {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .refresh-button {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: var(--border-radius);
            background-color: white;
            border: 1px solid var(--border-color);
            font-size: 0.875rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .refresh-button:hover {
            background-color: var(--gray-100);
        }

        .refresh-icon {
            font-size: 1rem;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        @media (min-width: 640px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (min-width: 1024px) {
            .stats-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        .stat-card-container {
            position: relative;
        }

        .glassmorphism {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: var(--border-radius-xl);
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .glassmorphism:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-0.25rem);
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .stat-value {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-tooltip {
            position: absolute;
            bottom: 100%;
            margin-bottom: 0.5rem;
            left: 50%;
            transform: translateX(-50%);
            width: max-content;
            padding: 0.25rem 0.5rem;
            background-color: var(--gray-800);
            color: white;
            font-size: 0.75rem;
            border-radius: 0.375rem;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }

        .stat-card-container:hover .stat-tooltip {
            opacity: 1;
        }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        @media (min-width: 1024px) {
            .charts-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        .chart-card {
            border-radius: var(--border-radius-xl);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .chart-container {
            flex-grow: 1;
            min-height: 250px;
        }

        /* Footer */
        .footer {
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border-color);
            font-size: 0.875rem;
            color: var(--text-secondary);
            background-color: white;
        }

        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .footer-links {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .footer-link {
            color: var(--text-secondary);
            text-decoration: none;
        }

        .footer-link:hover {
            color: var(--text-primary);
        }

        .language-selector {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .language-button {
            padding: 0.25rem 0.75rem;
            border-radius: 0.375rem;
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
        }

        .language-button:hover {
            background-color: var(--gray-100);
        }

        .language-divider {
            color: var(--border-color);
        }

        .social-links {
            display: flex;
            gap: 1rem;
        }

        .social-link {
            color: var(--text-secondary);
            transition: color 0.3s;
        }

        .social-link:hover {
            color: var(--primary-color);
        }

        .social-icon {
            width: 20px;
            height: 20px;
        }

        /* Additional Components */
        .card {
            background-color: white;
            border-radius: var(--border-radius-lg);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            padding: 1.5rem;
        }

        .card-header {
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .card-body {
            margin-bottom: 1rem;
        }

        .card-footer {
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem 1rem;
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .btn-primary {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        .btn-primary:hover {
            background-color: var(--primary-hover);
            border-color: var(--primary-hover);
        }

        .btn-secondary {
            background-color: var(--gray-200);
            color: var(--text-primary);
            border-color: var(--gray-200);
        }

        .btn-secondary:hover {
            background-color: var(--gray-300);
            border-color: var(--gray-300);
        }

        .btn-outline {
            background-color: transparent;
            color: var(--primary-color);
            border-color: var(--primary-color);
        }

        .btn-outline:hover {
            background-color: var(--primary-color);
            color: white;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .form-input,
        .form-select,
        .form-textarea {
            width: 100%;
            padding: 0.5rem 0.75rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            font-size: 0.875rem;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(13, 139, 242, 0.2);
        }

        .form-input::placeholder,
        .form-textarea::placeholder {
            color: var(--text-muted);
        }

        /* Dark Mode Toggle */
        .dark-mode-toggle {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            z-index: 50;
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            background-color: var(--primary-color);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: var(--shadow-lg);
            border: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
            }
            
            .sidebar.open {
                transform: translateX(0);
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 640px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .footer-content {
                flex-direction: column;
                text-align: center;
            }
            
            .footer-links {
                justify-content: center;
            }
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateY(10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .animate-fadeIn {
            animation: fadeIn 0.5s ease-out;
        }

        .animate-slideIn {
            animation: slideIn 0.5s ease-out;
        }

        /* Utility Classes */
        .text-center { text-align: center; }
        .text-left { text-align: left; }
        .text-right { text-align: right; }
        
        .hidden { display: none; }
        .block { display: block; }
        .inline-block { display: inline-block; }
        .flex { display: flex; }
        .inline-flex { display: inline-flex; }
        
        .justify-start { justify-content: flex-start; }
        .justify-end { justify-content: flex-end; }
        .justify-center { justify-content: center; }
        .justify-between { justify-content: space-between; }
        .justify-around { justify-content: space-around; }
        
        .items-start { align-items: flex-start; }
        .items-end { align-items: flex-end; }
        .items-center { align-items: center; }
        .items-baseline { align-items: baseline; }
        .items-stretch { align-items: stretch; }
        
        .flex-col { flex-direction: column; }
        .flex-row { flex-direction: row; }
        .flex-wrap { flex-wrap: wrap; }
        
        .gap-1 { gap: 0.25rem; }
        .gap-2 { gap: 0.5rem; }
        .gap-3 { gap: 0.75rem; }
        .gap-4 { gap: 1rem; }
        .gap-5 { gap: 1.25rem; }
        .gap-6 { gap: 1.5rem; }
        
        .m-1 { margin: 0.25rem; }
        .m-2 { margin: 0.5rem; }
        .m-3 { margin: 0.75rem; }
        .m-4 { margin: 1rem; }
        .m-5 { margin: 1.25rem; }
        .m-6 { margin: 1.5rem; }
        
        .p-1 { padding: 0.25rem; }
        .p-2 { padding: 0.5rem; }
        .p-3 { padding: 0.75rem; }
        .p-4 { padding: 1rem; }
        .p-5 { padding: 1.25rem; }
        .p-6 { padding: 1.5rem; }
        
        .mt-1 { margin-top: 0.25rem; }
        .mt-2 { margin-top: 0.5rem; }
        .mt-3 { margin-top: 0.75rem; }
        .mt-4 { margin-top: 1rem; }
        .mt-5 { margin-top: 1.25rem; }
        .mt-6 { margin-top: 1.5rem; }
        
        .mb-1 { margin-bottom: 0.25rem; }
        .mb-2 { margin-bottom: 0.5rem; }
        .mb-3 { margin-bottom: 0.75rem; }
        .mb-4 { margin-bottom: 1rem; }
        .mb-5 { margin-bottom: 1.25rem; }
        .mb-6 { margin-bottom: 1.5rem; }
        
        .rounded { border-radius: var(--border-radius); }
        .rounded-lg { border-radius: var(--border-radius-lg); }
        .rounded-xl { border-radius: var(--border-radius-xl); }
        .rounded-full { border-radius: 9999px; }
        
        .shadow { box-shadow: var(--shadow); }
        .shadow-lg { box-shadow: var(--shadow-lg); }
        .shadow-none { box-shadow: none; }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 42.4379C4 42.4379 14.0962 36.0744 24 41.1692C35.0664 46.8624 44 42.2078 44 42.2078L44 7.01134C44 7.01134 35.068 11.6577 24.0031 5.96913C14.0971 0.876274 4 7.27094 4 7.27094L4 42.4379Z" fill="currentColor"></path>
                    </svg>
                </div>
                <h2 class="sidebar-title">DIU Portal</h2>
            </div>
            
            <nav class="sidebar-nav">
                <a class="nav-link" href="#">
                    <span class="material-symbols-outlined nav-icon">chat</span>
                    <span class="nav-text">Chat</span>
                </a>
                <a class="nav-link active" href="#">
                    <span class="material-symbols-outlined nav-icon">dashboard</span>
                    <span class="nav-text">Dashboard</span>
                </a>
                <a class="nav-link" href="#">
                    <span class="material-symbols-outlined nav-icon">lightbulb</span>
                    <span class="nav-text">Recommendation</span>
                </a>
                <a class="nav-link" href="#">
                    <span class="material-symbols-outlined nav-icon">description</span>
                    <span class="nav-text">Application</span>
                </a>
                <a class="nav-link" href="#">
                    <span class="material-symbols-outlined nav-icon">calculate</span>
                    <span class="nav-text">Waiver Calculator</span>
                </a>
                <a class="nav-link" href="#">
                    <span class="material-symbols-outlined nav-icon">help</span>
                    <span class="nav-text">Help</span>
                </a>
            </nav>
            
            <div class="sidebar-footer">
                <div class="testimonial-box">
                    <p class="testimonial-text">"DIU changed my life. The community is amazing!"</p>
                    <p class="testimonial-author">- Alex Doe, CSE '23</p>
                </div>
                
                <button class="pref-button">
                    <span class="material-symbols-outlined">light_mode</span>
                    <span>Toggle Theme</span>
                </button>
                
                <button class="pref-button">
                    <span class="material-symbols-outlined">save</span>
                    <span>Save Preferences</span>
                </button>
            </div>
        </aside>
        
        <!-- Main Content -->
        <div class="main-content">
            <header class="header">
                <div class="header-left">
                    <button class="mobile-menu-button">
                        <span class="material-symbols-outlined">menu</span>
                    </button>
                    
                    <div class="search-container">
                        <span class="material-symbols-outlined search-icon">search</span>
                        <input class="search-input" placeholder="Search anything..." value="">
                    </div>
                </div>
                
                <div class="header-right">
                    <button class="header-button primary-button">
                        <span>Apply Now</span>
                    </button>
                    
                    <button class="header-button secondary-button">
                        <span>Chat with Us</span>
                    </button>
                    
                    <div class="user-avatar" style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuCvpNgaPdYm_JNIrOQw274_G3MkZlnMmfRdlnKotqHOaU1qea-L_wyOcVuTZkJRjxD_di4H7FJK7fbRbGfzXlCnmjOJDPVieccNkCT3wa2Iys2TDUfEmnGCW7dAuBHXTTnACiR_--vXjwdEy81i9sb47Z0dsAYQ0HCsmAti-vm73CDUo5_AgdO1GD_h4GsD2iFwro_fnVd7UwnAjJlfgJGnQrQERlRR_d6y8ZmEUPOdehnCZfL2SvIbe9HmUGeSUE2cu-JQrz5Hq8A");'></div>
                </div>
            </header>
            
            <main class="main-area">
                <div class="page-header">
                    <h1 class="page-title">Welcome, Sarah!</h1>
                    <button class="refresh-button">
                        <span class="material-symbols-outlined refresh-icon">refresh</span>
                        <span>Refresh Data</span>
                    </button>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card-container">
                        <div class="glassmorphism">
                            <p class="stat-label">Programs</p>
                            <p class="stat-value">12</p>
                        </div>
                        <div class="stat-tooltip">Click to view details</div>
                    </div>
                    
                    <div class="stat-card-container">
                        <div class="glassmorphism">
                            <p class="stat-label">Departments</p>
                            <p class="stat-value">3</p>
                        </div>
                        <div class="stat-tooltip">Click to view details</div>
                    </div>
                    
                    <div class="stat-card-container">
                        <div class="glassmorphism">
                            <p class="stat-label">Waivers</p>
                            <p class="stat-value">5</p>
                        </div>
                        <div class="stat-tooltip">Explore waiver options</div>
                    </div>
                    
                    <div class="stat-card-container">
                        <div class="glassmorphism">
                            <p class="stat-label">Applications</p>
                            <p class="stat-value">2</p>
                        </div>
                        <div class="stat-tooltip">Manage your applications</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-card glassmorphism">
                        <p class="chart-title">Applications by Program</p>
                        <div class="chart-container" id="bar-chart"></div>
                    </div>
                    
                    <div class="chart-card glassmorphism">
                        <p class="chart-title">Waiver Distribution</p>
                        <div class="chart-container" id="pie-chart"></div>
                    </div>
                    
                    <div class="chart-card glassmorphism">
                        <p class="chart-title">Application Trends</p>
                        <div class="chart-container" id="line-chart"></div>
                    </div>
                </div>
            </main>
            
            <footer class="footer">
                <div class="footer-content">
                    <p>© 2024 Daffodil International University. All rights reserved.</p>
                    
                    <div class="footer-links">
                        <a class="footer-link" href="#">About</a>
                        <a class="footer-link" href="#">Contact</a>
                        <a class="footer-link" href="#">Privacy</a>
                    </div>
                    
                    <div class="language-selector">
                        <button class="language-button">English</button>
                        <span class="language-divider">|</span>
                        <button class="language-button">বাংলা</button>
                    </div>
                    
                    <div class="social-links">
                        <a class="social-link" href="#">
                            <svg fill="currentColor" height="20px" viewBox="0 0 256 256" width="20px" xmlns="http://www.w3.org/2000/svg">
                                <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm8,191.63V152h24a8,8,0,0,0,0-16H136V112a16,16,0,0,1,16-16h16a8,8,0,0,0,0-16H152a32,32,0,0,0-32,32v24H96a8,8,0,0,0,0,16h24v63.63a88,88,0,1,1,16,0Z"></path>
                            </svg>
                        </a>
                        
                        <a class="social-link" href="#">
                            <svg fill="currentColor" height="20px" viewBox="0 0 256 256" width="20px" xmlns="http://www.w3.org/2000/svg">
                                <path d="M247.39,68.94A8,8,0,0,0,240,64H209.57A48.66,48.66,0,0,0,168.1,40a46.91,46.91,0,0,0-33.75,13.7A47.9,47.9,0,0,0,120,88v6.09C79.74,83.47,46.81,50.72,46.46,50.37a8,8,0,0,0-13.65,4.92c-4.31,47.79,9.57,79.77,22,98.18a110.93,110.93,0,0,0,21.88,24.2c-15.23,17.53-39.21,26.74-39.47,26.84a8,8,0,0,0-3.85,11.93c.75,1.12,3.75,5.05,11.08,8.72C53.51,229.7,65.48,232,80,232c70.67,0,129.72-54.42,135.75-124.44l29.91-29.9A8,8,0,0,0,247.39,68.94Zm-45,29.41a8,8,0,0,0-2.32,5.14C196,166.58,143.28,216,80,216c-10.56,0-18-1.4-23.22-3.08,11.51-6.25,27.56-17,37.88-32.48A8,8,0,0,0,92,169.08c-.47-.27-43.91-26.34-44-96,16,13,45.25,33.17,78.67,38.79A8,8,0,0,0,136,104V88a32,32,0,0,1,9.6-22.92A30.94,30.94,0,0,1,167.9,56c12.66.16,24.49,7.88,29.44,19.21A8,8,0,0,0,204.67,80h16Z"></path>
                            </svg>
                        </a>
                        
                        <a class="social-link" href="#">
                            <svg fill="currentColor" height="20px" viewBox="0 0 256 256" width="20px" xmlns="http://www.w3.org/2000/svg">
                                <path d="M128,80a48,48,0,1,0,48,48A48.05,48.05,0,0,0,128,80Zm0,80a32,32,0,1,1,32-32A32,32,0,0,1,128,160ZM176,24H80A56.06,56.06,0,0,0,24,80v96a56.06,56.06,0,0,0,56,56h96a56.06,56.06,0,0,0,56-56V80A56.06,56.06,0,0,0,176,24Zm40,152a40,40,0,0,1-40,40H80a40,40,0,0,1-40-40V80A40,40,0,0,1,80,40h96a40,40,0,0,1,40,40ZM192,76a12,12,0,1,1-12-12A12,12,0,0,1,192,76Z"></path>
                            </svg>
                        </a>
                    </div>
                </div>
            </footer>
        </div>
    </div>
    
    <button class="dark-mode-toggle">
        <span class="material-symbols-outlined">dark_mode</span>
    </button>

    <script>
        // Toggle mobile menu
        document.querySelector('.mobile-menu-button').addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('open');
        });
        
        // Toggle dark mode
        document.querySelector('.dark-mode-toggle').addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
        });
        
        // Toggle theme from sidebar
        document.querySelector('.pref-button:first-child').addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
        });
    </script>

</style>
""", unsafe_allow_html=True)

# ============================
# 2. Data Loading with Enhanced Debugging
# ============================
@st.cache_data
def load_faq_data(csv_path):
    full_path = os.path.join(BASE_DIR, csv_path)
    logger.info(f"Attempting to load FAQ data from: {full_path}")
    try:
        if not os.path.exists(full_path):
            logger.warning(f"FAQ file not found at {full_path}. Using default data.")
            return create_default_faq()
        df = pd.read_csv(full_path, quotechar='"', on_bad_lines='warn', engine='python', encoding='utf-8')
        logger.info(f"Raw CSV columns: {df.columns.tolist()}")
        logger.info(f"Total rows before filtering: {len(df)}")
        logger.info(f"First row sample: {df.iloc[0].to_dict()}")
        required_cols = ['question', 'answer', 'category']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        missing_rows = df[df[required_cols].isna().any(axis=1)]
        if not missing_rows.empty:
            logger.warning(f"Found {len(missing_rows)} rows with missing values:\n{missing_rows}")
            df[required_cols] = df[required_cols].fillna('[MISSING]')
        df['question'] = df['question'].astype(str)
        df['answer'] = df['answer'].astype(str)
        df['category'] = df['category'].astype(str)
        logger.info(f"Successfully loaded FAQ data with {len(df)} entries.")
        return df
    except Exception as e:
        logger.error(f"Error loading FAQ from {full_path}: {str(e)}. Using default data.")
        st.error(f"Error loading FAQ from {full_path}: {str(e)}. Using default data.")
        return create_default_faq()

@st.cache_data
def load_json(file_path):
    full_path = os.path.join(BASE_DIR, file_path)
    logger.info(f"Attempting to load JSON data from: {full_path}")
    try:
        if not os.path.exists(full_path):
            logger.warning(f"JSON file not found at {full_path}. Creating empty file.")
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            return []
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                logger.warning(f"{full_path} is empty. Using default data.")
                return []
            data = json.loads(content)
            logger.info(f"Successfully loaded JSON data with {len(data)} entries.")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Error loading {full_path}: Invalid JSON format ({str(e)}). Using default data.")
        return []
    except Exception as e:
        logger.error(f"Error loading {full_path}: {str(e)}. Using default data.")
        return []

def create_default_faq():
    return pd.DataFrame({
        "question": [
            "What are the admission requirements?",
            "How do I apply for admission?",
            "What programs does DIU offer?"
        ],
        "answer": [
            "Minimum GPA of 2.5 in both SSC and HSC with a total GPA of 6.0.",
            "Apply online through our portal or visit our admission office.",
            "DIU offers programs in Engineering, Business, Humanities, and more."
        ],
        "category": ["General"] * 3
    })

def create_sample_data():
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    faq_path = os.path.join(data_dir, "faq.csv")
    if not os.path.exists(faq_path):
        faq_data = {
            "SL": range(1, 10),
            "Question ID": [f"Q{i:06d}" for i in range(100001, 100010)],
            "question": [
                "How do I apply for admission?",
                "What are the admission requirements for undergraduate programs?",
                "What are the admission requirements for graduate programs?",
                "Is there an entrance test?",
                "Can I apply without my HSC transcript?",
                "When is the application deadline?",
                "Can I change my program after admission?",
                "Is there an age limit for admission?",
                "Can international students apply?"
            ],
            "answer": [
                "You can apply online through the Daffodil International University admission portal or submit a printed form at the admissions office.",
                "Students must have completed HSC or equivalent with a minimum GPA requirement set by the university.",
                "Applicants must hold a recognized bachelor's degree and meet the GPA and departmental requirements.",
                "Yes, some programs require an entrance test. Check your department for details.",
                "No, HSC transcript is mandatory for undergraduate admission.",
                "Deadlines vary by program. Generally, the Fall semester deadline is in July and Spring semester is in December.",
                "Yes, with departmental approval, students may change their program within the first month of semester.",
                "No specific age limit exists, but applicants must meet academic requirements.",
                "Yes, international students can apply. They must submit equivalent qualifications and English proficiency documents."
            ],
            "category": ["Admission Requirements & Process"] * 9,
            "Source link": ["https://daffodilvarsity.edu.bd/admission"] * 9,
            "keywords": [
                "apply admission",
                "undergraduate admission",
                "graduate admission",
                "entrance test",
                "HSC transcript required",
                "application deadline",
                "change program",
                "age limit",
                "international admission"
            ]
        }
        pd.DataFrame(faq_data).to_csv(faq_path, index=False)
        logger.info(f"Created sample FAQ data at {faq_path}")

    # Simplified waivers, programs, departments for brevity
    waivers_path = os.path.join(data_dir, "waivers.json")
    if not os.path.exists(waivers_path):
        with open(waivers_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    programs_path = os.path.join(data_dir, "programs.json")
    if not os.path.exists(programs_path):
        with open(programs_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    departments_path = os.path.join(data_dir, "departments.json")
    if not os.path.exists(departments_path):
        with open(departments_path, "w", encoding="utf-8") as f:
            json.dump([], f)

# Load data
create_sample_data()
faq_df = load_faq_data(os.path.join("data", "faq.csv"))
waivers = load_json(os.path.join("data", "waivers.json"))
programs = load_json(os.path.join("data", "programs.json"))
departments = load_json(os.path.join("data", "departments.json"))

# ============================
# 3. Personalization
# ============================
if "user_prefs" not in st.session_state:
    st.session_state.user_prefs = {
        "theme": "light",
        "layout": "wide",
        "notifications": True
    }

if "application_progress" not in st.session_state:
    st.session_state.application_progress = {}

if "current_step" not in st.session_state:
    st.session_state.current_step = 1

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

if "submitted_applications" not in st.session_state:
    st.session_state.submitted_applications = []

# ============================
# 4. FAQ Matching
# ============================
def get_faq_answer(user_input, faq_df, threshold=0.2):
    if faq_df.empty:
        logger.warning("FAQ DataFrame is empty.")
        return "I'm sorry, I don't have enough information to answer your question."

    logger.info(f"DataFrame columns: {list(faq_df.columns)}")
    logger.info(f"First row sample: {faq_df.iloc[0].to_dict()}")

    if 'answer' not in faq_df.columns or 'category' not in faq_df.columns:
        logger.error(f"Required columns missing. Available columns: {faq_df.columns}")
        return "Error: FAQ data is missing required columns. Please check the CSV file."

    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                      'what', 'how', 'when', 'where', 'why', 'of', 'in', 'on', 'to'])
    user_words = [word for word in user_input.lower().split() if word not in stop_words]
    processed_input = ' '.join(user_words)

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(faq_df['question'])
        user_vec = vectorizer.transform([processed_input])
        similarity = cosine_similarity(user_vec, tfidf_matrix)
        best_idx = similarity.argmax()

        if similarity[0, best_idx] >= threshold:
            matched_answer = faq_df.iloc[best_idx]['answer']
            matched_category = faq_df.iloc[best_idx]['category']
            logger.info(f"TF-IDF match: Question='{faq_df.iloc[best_idx]['question']}', "
                        f"Similarity={similarity[0, best_idx]:.4f}, "
                        f"Answer='{matched_answer}', Category='{matched_category}'")
            return matched_answer
    except Exception as e:
        logger.error(f"TF-IDF error: {str(e)}")

    user_input_lower = user_input.lower().strip()
    user_keywords = set(user_input_lower.split())
    best_match_score = 0
    best_match_index = -1
    min_score_threshold = 2

    for idx, row in faq_df.iterrows():
        question_lower = row['question'].lower()
        if 'http' in question_lower or 'www.' in question_lower:
            continue
        keywords_lower = row['keywords'].lower().split(',') if pd.notna(row['keywords']) else []
        all_terms = set(question_lower.split() + keywords_lower)
        match_score = len(user_keywords.intersection(all_terms))
        boosts = ["bba", "nfe", "admission test"]
        for phrase in boosts:
            if phrase in user_input_lower and phrase in question_lower:
                match_score += 2
        if match_score > best_match_score and match_score >= min_score_threshold:
            best_match_score = match_score
            best_match_index = idx

    if best_match_index != -1 and best_match_score > 0:
        matched_answer = faq_df.iloc[best_match_index]['answer']
        matched_category = faq_df.iloc[best_match_index]['category']
        logger.info(f"Keyword match: Question='{faq_df.iloc[best_match_index]['question']}', "
                    f"Score={best_match_score}, Answer='{matched_answer}', Category='{matched_category}'")
        return matched_answer

    logger.warning(f"No match for: {user_input}")
    return "I'm sorry, I couldn't find an answer to your question. Please try rephrasing or contact admission@diu.net.bd."

# ============================
# 5. Department Recommendation
# ============================
def recommend_department(user_interests, gpa_hsc, gpa_ssc):
    recommended = []
    for dept in departments:
        score = 0
        for tag in dept.get("tags", []):
            if isinstance(tag, str) and tag.lower() in user_interests.lower():
                score += 1
        min_gpa = dept.get("min_gpa", 0)
        avg_gpa = (gpa_hsc + gpa_ssc) / 2
        if avg_gpa >= min_gpa:
            score += 1
        if score > 0:
            recommended.append({"name": dept["name"], "score": score, "details": dept.get("details", "")})
    return sorted(recommended, key=lambda x: x["score"], reverse=True)[:3]

# ============================
# 6. DIU WAIVER CALCULATION SYSTEM
# ============================
class DIUWaiverCalculator:
    def __init__(self):
        self.waiver_data = self._load_waiver_data()
    
    def _load_waiver_data(self):
        return {
            "result_based": {
                "SIT_BE_AHS_Engineering": [
                    {"condition": "Golden GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 75, "sgpa_req": 3.5, "for_new_students": True},
                    {"condition": "Golden GPA-5 in HSC", "min_hsc": 5.0, "waiver": 50, "sgpa_req": 3.25, "for_new_students": True},
                    {"condition": "GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 35, "sgpa_req": 3.25, "for_new_students": True},
                    {"condition": "GPA-5 in HSC", "min_hsc": 5.0, "waiver": 25, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "HSC GPA 4.90-4.99", "min_hsc": 4.9, "max_hsc": 4.99, "waiver": 20, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "HSC GPA 4.75-4.89", "min_hsc": 4.75, "max_hsc": 4.89, "waiver": 10, "sgpa_req": 3.0, "for_new_students": True}
                ],
                "Humanities_Social_Sciences": [
                    {"condition": "Golden GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 75, "sgpa_req": 3.5, "for_new_students": True},
                    {"condition": "Golden GPA-5 in HSC", "min_hsc": 5.0, "waiver": 50, "sgpa_req": 3.25, "for_new_students": True},
                    {"condition": "GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 35, "sgpa_req": 3.25, "for_new_students": True},
                    {"condition": "GPA-5 in HSC", "min_hsc": 5.0, "waiver": 25, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "HSC GPA 4.90-4.99", "min_hsc": 4.9, "max_hsc": 4.99, "waiver": 20, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "HSC GPA 4.80-4.89", "min_hsc": 4.8, "max_hsc": 4.89, "waiver": 15, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "HSC GPA 4.50-4.79", "min_hsc": 4.5, "max_hsc": 4.79, "waiver": 10, "sgpa_req": 3.0, "for_new_students": True}
                ],
                "BPharm_LLB_CSE": [
                    {"condition": "Golden GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 50, "sgpa_req": 3.25, "for_new_students": True},
                    {"condition": "Golden GPA-5 in HSC", "min_hsc": 5.0, "waiver": 30, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "GPA-5 both in SSC and HSC", "min_ssc": 5.0, "min_hsc": 5.0, "waiver": 25, "sgpa_req": 3.0, "for_new_students": True},
                    {"condition": "GPA-5 in HSC", "min_hsc": 5.0, "waiver": 20, "sgpa_req": 3.0, "for_new_students": True}
                ]
            },
            "sgpa_based": {
                "BE_SIT_AHS_Engineering": [
                    {"gpa_range": "4.00", "waiver": 50, "for_new_students": False},
                    {"gpa_range": "3.90-3.99", "waiver": 30, "for_new_students": False},
                    {"gpa_range": "3.85-3.89", "waiver": 20, "for_new_students": False},
                    {"gpa_range": "3.80-3.84", "waiver": 10, "for_new_students": False}
                ],
                "Humanities_Social_Sciences": [
                    {"gpa_range": "3.90+", "waiver": 50, "for_new_students": False},
                    {"gpa_range": "3.85-3.89", "waiver": 40, "for_new_students": False},
                    {"gpa_range": "3.80-3.84", "waiver": 20, "for_new_students": False},
                    {"gpa_range": "3.75-3.79", "waiver": 15, "for_new_students": False},
                    {"gpa_range": "3.60-3.74", "waiver": 10, "for_new_students": False}
                ]
            },
            "special_quotas": {
                "female": {
                    "SIT_BE_AHS_Engineering": {"min_hsc": 4.0, "max_hsc": 4.74, "waiver": 10, "sgpa_req": 3.0, "for_new_students": True},
                    "Humanities_Social_Sciences": {"min_hsc": 4.0, "max_hsc": 4.49, "waiver": 10, "sgpa_req": 3.0, "for_new_students": True}
                },
                "diu_employee": {"waiver": 50, "sgpa_req": 3.0, "for_new_students": True},
                "dic_student": {"waiver": 20, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "dpi_student": {"waiver": 20, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "dipti_student": {
                    "result_worse": {"waiver": 15, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                    "result_better": {"waiver": 25, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True}
                },
                "alumni_relative": {"waiver": 10, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "alumni_spouse": {"waiver": 10, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "physically_challenged": {"waiver": 25, "sgpa_req": 2.5, "min_credits": 12, "for_new_students": True},
                "tribal": {"waiver": 15, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "sibling_spouse": {"waiver": 20, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                "diploma_holder": [
                    {"gpa_range": "3.90-4.00", "waiver": 75, "sgpa_req": 3.5, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "3.80-3.89", "waiver": 60, "sgpa_req": 3.5, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "3.75-3.79", "waiver": 50, "sgpa_req": 3.25, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "3.50-3.74", "waiver": 40, "sgpa_req": 3.25, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "3.25-3.49", "waiver": 30, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "3.00-3.24", "waiver": 25, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True},
                    {"gpa_range": "2.50-2.99", "waiver": 15, "sgpa_req": 3.0, "min_credits": 18, "for_new_students": True}
                ],
                "First Batch": {"waiver": 15, "sgpa_req": 3.0, "for_new_students": True},
                "Player": {
                    "National Team": {"waiver": 100, "sgpa_req": 2.0, "min_credits_ug": 6, "min_credits_masters": 6, "for_new_students": True},
                    "Premier League": {"waiver": 90, "sgpa_req": 2.0, "min_credits_ug": 12, "min_credits_masters": 9, "for_new_students": True},
                    "First Division": {"waiver": 60, "sgpa_req": 2.0, "min_credits_ug": 12, "min_credits_masters": 9, "for_new_students": True},
                    "Second Division": {"waiver": 40, "sgpa_req": 2.0, "min_credits_ug": 12, "min_credits_masters": 9, "for_new_students": True},
                    "DIU Player": {"waiver_range": "20-40", "sgpa_req": 2.0, "min_credits_ug": 12, "min_credits_masters": 9, "for_new_students": True}
                }
            }
        }
    
    def calculate_result_based_waivers(self, faculty, ssc_gpa, hsc_gpa, is_new_student=True):
        eligible = []
        
        if faculty not in self.waiver_data["result_based"]:
            return eligible
            
        for waiver in self.waiver_data["result_based"][faculty]:
            if not waiver.get("for_new_students", True) and is_new_student:
                continue
                
            meets_condition = True
            
            if "min_ssc" in waiver and ssc_gpa < waiver["min_ssc"]:
                meets_condition = False
            if "min_hsc" in waiver and hsc_gpa < waiver["min_hsc"]:
                meets_condition = False
            if "max_hsc" in waiver and hsc_gpa > waiver["max_hsc"]:
                meets_condition = False
                
            if meets_condition:
                requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {waiver['sgpa_req']}"
                
                eligible.append({
                    "type": "Result-based",
                    "condition": waiver["condition"],
                    "waiver_percentage": waiver["waiver"],
                    "requirements": requirements,
                    "for_new_students": is_new_student
                })
                
        return eligible
    
    def calculate_sgpa_based_waivers(self, faculty, current_sgpa, is_new_student=True):
        eligible = []
        
        if is_new_student or faculty not in self.waiver_data["sgpa_based"]:
            return eligible
            
        for waiver in self.waiver_data["sgpa_based"][faculty]:
            gpa_range = waiver["gpa_range"]
            
            if gpa_range == "4.00" and current_sgpa == 4.0:
                eligible.append({
                    "type": "SGPA-based",
                    "condition": "Perfect 4.0 SGPA",
                    "waiver_percentage": waiver["waiver"],
                    "requirements": "Maintain excellent academic performance",
                    "for_new_students": False
                })
            elif "+" in gpa_range:
                min_gpa = float(gpa_range.replace("+", ""))
                if current_sgpa >= min_gpa:
                    eligible.append({
                        "type": "SGPA-based",
                        "condition": f"SGPA {gpa_range}",
                        "waiver_percentage": waiver["waiver"],
                        "requirements": "Maintain excellent academic performance",
                        "for_new_students": False
                    })
            elif "-" in gpa_range:
                min_gpa, max_gpa = map(float, gpa_range.split("-"))
                if min_gpa <= current_sgpa <= max_gpa:
                    eligible.append({
                        "type": "SGPA-based",
                        "condition": f"SGPA {gpa_range}",
                        "waiver_percentage": waiver["waiver"],
                        "requirements": "Maintain good academic performance",
                        "for_new_students": False
                    })
                    
        return eligible
    
    def calculate_special_quota_waivers(self, quota_type, faculty=None, hsc_gpa=None, 
                                      is_new_student=True, current_sgpa=0, **kwargs):
        eligible = []
        
        if quota_type not in self.waiver_data["special_quotas"]:
            return eligible
            
        quota_data = self.waiver_data["special_quotas"][quota_type]
        
        if quota_type == "female":
            if faculty in quota_data:
                criteria = quota_data[faculty]
                if (criteria["min_hsc"] <= hsc_gpa <= criteria["max_hsc"]):
                    requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {criteria['sgpa_req']}"
                    
                    eligible.append({
                        "type": "Female Quota",
                        "condition": f"Female student with HSC GPA {hsc_gpa}",
                        "waiver_percentage": criteria["waiver"],
                        "requirements": requirements,
                        "for_new_students": is_new_student
                    })
                    
        elif quota_type == "dipti_student":
            is_better = kwargs.get("hsc_better_than_ssc", False)
            criteria = quota_data["result_better"] if is_better else quota_data["result_worse"]
            
            requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {criteria['sgpa_req']}, Take {criteria['min_credits']} credits"
                
            eligible.append({
                "type": "DIPTI Student Quota",
                "condition": f"DIPTI student with {'better' if is_better else 'same/worse'} HSC result",
                "waiver_percentage": criteria["waiver"],
                "requirements": requirements,
                "for_new_students": is_new_student
            })
                
        elif quota_type == "diploma_holder":
            diploma_gpa = kwargs.get("diploma_gpa", 0)
            for waiver in quota_data:
                if "-" in waiver["gpa_range"]:
                    min_gpa, max_gpa = map(float, waiver["gpa_range"].split("-"))
                    if min_gpa <= diploma_gpa <= max_gpa:
                        requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {waiver['sgpa_req']}, Take {waiver['min_credits']} credits"
                        
                        eligible.append({
                            "type": "Diploma Holder Quota",
                            "condition": f"Diploma GPA {diploma_gpa}",
                            "waiver_percentage": waiver["waiver"],
                            "requirements": requirements,
                            "for_new_students": is_new_student
                        })
                        
        elif quota_type == "player":
            player_level = kwargs.get("player_level", "").lower()
            if player_level in quota_data:
                criteria = quota_data[player_level]
                requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {criteria['sgpa_req']}, Take {criteria['min_credits_ug']} credits (UG) or {criteria['min_credits_masters']} credits (Masters)"
                    
                eligible.append({
                    "type": f"{player_level.title()} Player Quota",
                    "condition": f"Recognized {player_level} level player",
                    "waiver_percentage": criteria["waiver"],
                    "requirements": requirements,
                    "for_new_students": is_new_student
                })
                    
        else:
            sgpa_req = quota_data.get("sgpa_req", 0)
            requirements = "Maintain SGPA after admission" if is_new_student else f"Maintain SGPA: {sgpa_req}"
            if 'min_credits' in quota_data:
                requirements += f", Take {quota_data['min_credits']} credits"
                
            eligible.append({
                "type": f"{quota_type.replace('_', ' ').title()} Quota",
                "condition": f"Eligible for {quota_type.replace('_', ' ')} quota",
                "waiver_percentage": quota_data["waiver"],
                "requirements": requirements,
                "for_new_students": is_new_student
            })
                
        return eligible
    
    def calculate_comprehensive_waivers(self, faculty, ssc_gpa, hsc_gpa, is_new_student=True, current_sgpa=0, student_profile=None):
        if student_profile is None:
            student_profile = {}
            
        all_waivers = []
        
        all_waivers.extend(self.calculate_result_based_waivers(faculty, ssc_gpa, hsc_gpa, is_new_student))
        
        if not is_new_student:
            all_waivers.extend(self.calculate_sgpa_based_waivers(faculty, current_sgpa, is_new_student))
        
        if student_profile.get("is_female", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "female", faculty, hsc_gpa, is_new_student, current_sgpa
            ))
            
        if student_profile.get("is_diu_employee", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "diu_employee", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("is_dic_student", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "dic_student", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("is_dipti_student", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "dipti_student", 
                current_sgpa=current_sgpa,
                hsc_better_than_ssc=student_profile.get("hsc_better_than_ssc", False)
            ))
            
        if student_profile.get("is_alumni_relative", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "alumni_relative", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("is_physically_challenged", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "physically_challenged", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("is_tribal", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "tribal", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("has_sibling_student", False) or student_profile.get("has_spouse_student", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "sibling_spouse", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("is_diploma_holder", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "diploma_holder", 
                current_sgpa=current_sgpa,
                diploma_gpa=student_profile.get("diploma_gpa", 0)
            ))
            
        if student_profile.get("is_first_batch", False):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "first_batch", current_sgpa=current_sgpa
            ))
            
        if student_profile.get("player_level"):
            all_waivers.extend(self.calculate_special_quota_waivers(
                "player", 
                current_sgpa=current_sgpa,
                player_level=student_profile.get("player_level")
            ))
            
        return all_waivers

# Initialize the waiver calculator
waiver_calculator = DIUWaiverCalculator()

def calculate_waivers(hsc_gpa, ssc_gpa, faculty, is_new_student=True, current_sgpa=0, student_profile=None):
    if student_profile is None:
        student_profile = {}
        
    return waiver_calculator.calculate_comprehensive_waivers(
        faculty, ssc_gpa, hsc_gpa, is_new_student, current_sgpa, student_profile
    )
    
# ============================
# 7. Bot Response
# ============================
def get_bot_response(user_input):
    user_input_lower = user_input.lower().strip()
    logger.info(f"Processing user input: {user_input_lower}")

    if any(term in user_input_lower for term in ["waiver", "scholarship", "financial aid", "discount"]):
        return "I can help you understand DIU's waiver system! Use our Waiver Calculator tool to see what you might qualify for based on your academic performance and profile."
    
    if user_input_lower in ["hello", "hi", "hey"]:
        return "Welcome to DIU's Premium Admission Portal! Ask about admissions, programs, or waivers."
    if "how are you" in user_input_lower:
        return "I'm doing great! Ready to assist with your admission queries."
    if "help" in user_input_lower or "confused" in user_input_lower:
        return "Let's get you sorted! Try the 'Recommendation' tool or ask about specific programs."
    if "department" in user_input_lower or "choose" in user_input_lower:
        return "Use our 'Recommendation' tool to find the perfect department based on your GPA and interests."

    faq_answer = get_faq_answer(user_input, faq_df)
    if faq_answer and not faq_answer.startswith("Error:") and faq_answer != "I'm sorry, I couldn't find an answer to your question. Please try rephrasing or contact admission@diu.net.bd.":
        logger.info(f"Returning FAQ answer: {faq_answer}")
        return faq_answer

    for waiver in waivers:
        if isinstance(waiver.get("name"), str) and waiver["name"].lower() in user_input_lower:
            waiver_rate = waiver['waiver_rate'] if isinstance(waiver['waiver_rate'], str) else ", ".join(waiver['waiver_rate'])
            logger.info(f"Returning waiver response: {waiver['name']}")
            return f"**{waiver['name']}** ({waiver_rate}): {waiver['description']}."
    for program in programs:
        if isinstance(program.get("name"), str) and program["name"].lower() in user_input_lower:
            logger.info(f"Returning program response: {program['name']}")
            return f"**{program['name']}**: {program['details']}."

    logger.warning(f"No specific match found, returning default response")
    return "Could you rephrase your question? Try asking about programs or waivers."

# ============================
# 8. Enhanced Application Form Functions
# ============================
def validate_email(email):
    pattern = r'^[a-zA-Z00-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    phone = ''.join(filter(str.isdigit, phone))
    return len(phone) in [10, 11]

def validate_nid(nid):
    nid = ''.join(filter(str.isdigit, nid))
    return len(nid) >= 10

def save_application_data(application_data):
    application_id = f"APP{int(time.time())}"
    application_data["application_id"] = application_id
    application_data["submission_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.submitted_applications.append(application_data)
    
    try:
        with open("applications.json", "w") as f:
            json.dump(st.session_state.submitted_applications, f, indent=4)
    except:
        pass

def load_application_data():
    try:
        with open("applications.json", "r") as f:
            st.session_state.submitted_applications = json.load(f)
    except:
        pass

def export_applications():
    if not st.session_state.submitted_applications:
        return None
    
    df = pd.DataFrame(st.session_state.submitted_applications)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Applications', index=False)
    return output.getvalue()

def step1_personal_info():
    st.markdown('<div class="sub-header">Personal Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", value=st.session_state.form_data.get("full_name", ""), key="full_name")
        st.session_state.form_data["full_name"] = full_name
        
        father_name = st.text_input("Father's Name", value=st.session_state.form_data.get("father_name", ""), key="father_name")
        st.session_state.form_data["father_name"] = father_name
        
        mother_name = st.text_input("Mother's Name", value=st.session_state.form_data.get("mother_name", ""), key="mother_name")
        st.session_state.form_data["mother_name"] = mother_name
        
        dob = st.date_input("Date of Birth", value=st.session_state.form_data.get("dob", datetime(2000, 1, 1)), 
                           min_value=datetime(1980, 1, 1), max_value=datetime.today(), key="dob")
        st.session_state.form_data["dob"] = dob
    
    with col2:
        email = st.text_input("Email Address", value=st.session_state.form_data.get("email", ""), key="email")
        st.session_state.form_data["email"] = email
        
        phone = st.text_input("Phone Number", value=st.session_state.form_data.get("phone", ""), key="phone")
        st.session_state.form_data["phone"] = phone
        
        nid = st.text_input("National ID", value=st.session_state.form_data.get("nid", ""), key="nid")
        st.session_state.form_data["nid"] = nid
        
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                             index=["Male", "Female", "Other"].index(st.session_state.form_data.get("gender", "Male")), key="gender")
        st.session_state.form_data["gender"] = gender

def step2_academic_info():
    st.markdown('<div class="sub-header">Academic Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Secondary School Certificate (SSC)")
        ssc_gpa = st.number_input("SSC GPA", 0.0, 5.0, 
                                 value=float(st.session_state.form_data.get("ssc_gpa", 0.0)), 
                                 step=0.01, format="%.2f", key="ssc_gpa")
        st.session_state.form_data["ssc_gpa"] = ssc_gpa
        
        ssc_year = st.number_input("SSC Passing Year", 1990, 2025, 
                                  value=int(st.session_state.form_data.get("ssc_year", 2020)), key="ssc_year")
        st.session_state.form_data["ssc_year"] = ssc_year
        
        ssc_board = st.selectbox("SSC Board", ["Dhaka", "Rajshahi", "Comilla", "Chittagong", "Barisal", "Sylhet", "Dinajpur", "Jessore", "Mymensingh", "Madrasah"],
                                index=["Dhaka", "Rajshahi", "Comilla", "Chittagong", "Barisal", "Sylhet", "Dinajpur", "Jessore", "Mymensingh", "Madrasah"].index(
                                    st.session_state.form_data.get("ssc_board", "Dhaka")), key="ssc_board")
        st.session_state.form_data["ssc_board"] = ssc_board
        
        ssc_group = st.selectbox("SSC Group", ["Science", "Arts", "Commerce", "Vocational"],
                                index=["Science", "Arts", "Commerce", "Vocational"].index(
                                    st.session_state.form_data.get("ssc_group", "Science")), key="ssc_group")
        st.session_state.form_data["ssc_group"] = ssc_group
    
    with col2:
        st.markdown("##### Higher Secondary Certificate (HSC)")
        hsc_gpa = st.number_input("HSC GPA", 0.0, 5.0, 
                                 value=float(st.session_state.form_data.get("hsc_gpa", 0.0)), 
                                 step=0.01, format="%.2f", key="hsc_gpa")
        st.session_state.form_data["hsc_gpa"] = hsc_gpa
        
        hsc_year = st.number_input("HSC Passing Year", 1990, 2025, 
                                  value=int(st.session_state.form_data.get("hsc_year", 2022)), key="hsc_year")
        st.session_state.form_data["hsc_year"] = hsc_year
        
        hsc_board = st.selectbox("HSC Board", ["Dhaka", "Rajshahi", "Comilla", "Chittagong", "Barisal", "Sylhet", "Dinajpur", "Jessore", "Mymensingh", "Madrasah"],
                                index=["Dhaka", "Rajshahi", "Comilla", "Chittagong", "Barisal", "Sylhet", "Dinajpur", "Jessore", "Mymensingh", "Madrasah].index(
                                    st.session_state.form_data.get("hsc_board", "Dhaka")), key="hsc_board")
        st.session_state.form_data["hsc_board"] = hsc_board
        
        hsc_group = st.selectbox("HSC Group", ["Science", "Arts", "Commerce", "Vocational"],
                                index=["Science", "Arts", "Commerce", "Vocational"].index(
                                    st.session_state.form_data.get("hsc_group", "Science")), key="hsc_group")
        st.session_state.form_data["hsc_group"] = hsc_group

def step3_program_selection():
    st.markdown('<div class="sub-header">Program Selection</div>', unsafe_allow_html=True)
    
    st.markdown("### Available Programs")
    for program in programs:
        with st.container():
            st.markdown(f'''
            <div class="program-card">
                <div class="program-title">{program["name"]} ({program["code"]})</div>
                <div class="program-details">
                    <span class="program-detail">⏱️ {program["duration"]}</span>
                    <span class="program-detail">📊 {program["credits"]} credits</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        program_choice = st.selectbox("Preferred Program", [p["name"] for p in programs], 
                                     index=[p["name"] for p in programs].index(
                                         st.session_state.form_data.get("program_choice", programs[0]["name"])), key="program_choice")
        st.session_state.form_data["program_choice"] = program_choice
        
        program_code = next((p["code"] for p in programs if p["name"] == program_choice), "")
        st.session_state.form_data["program_code"] = program_code
    
    with col2:
        semester = st.selectbox("Preferred Semester", ["Spring", "Summer", "Fall"], 
                               index=["Spring", "Summer", "Fall"].index(
                                   st.session_state.form_data.get("semester", "Spring")), key="semester")
        st.session_state.form_data["semester"] = semester
        
        year = st.number_input("Year", datetime.now().year, datetime.now().year + 5, 
                              value=int(st.session_state.form_data.get("year", datetime.now().year)), key="year")
        st.session_state.form_data["year"] = year

def step4_documents():
    st.markdown('<div class="sub-header">Documents Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Required Documents")
        photo = st.file_uploader("Upload Photo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="photo")
        if photo:
            st.session_state.form_data["photo"] = photo.name
            st.info(f"Uploaded: {photo.name}")
        
        ssc_cert = st.file_uploader("Upload SSC Certificate (PDF, JPG)", type=["pdf", "jpg", "jpeg"], key="ssc_cert")
        if ssc_cert:
            st.session_state.form_data["ssc_cert"] = ssc_cert.name
            st.info(f"Uploaded: {ssc_cert.name}")
    
    with col2:
        st.markdown("##### Additional Documents (Optional)")
        hsc_cert = st.file_uploader("Upload HSC Certificate (PDF, JPG)", type=["pdf", "jpg", "jpeg"], key="hsc_cert")
        if hsc_cert:
            st.session_state.form_data["hsc_cert"] = hsc_cert.name
            st.info(f"Uploaded: {hsc_cert.name}")
        
        nid_copy = st.file_uploader("Upload NID Copy (PDF, JPG)", type["pdf", "jpg", "jpeg"], key="nid_copy")
        if nid_copy:
            st.session_state.form_data["nid_copy"] = nid_copy.name
            st.info(f"Uploaded: {nid_copy.name}")

def step5_review():
    st.markdown('<div class="sub-header">Review Your Application</div>', unsafe_allow_html=True)
    
    st.markdown("### Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Full Name:** {st.session_state.form_data.get('full_name', '')}")
        st.markdown(f"**Father's Name:** {st.session_state.form_data.get('father_name', '')}")
        st.markdown(f"**Mother's Name:** {st.session_state.form_data.get('mother_name', '')}")
        st.markdown(f"**Date of Birth:** {st.session_state.form_data.get('dob', '')}")
    
    with col2:
        st.markdown(f"**Email:** {st.session_state.form_data.get('email', '')}")
        st.markdown(f"**Phone:** {st.session_state.form_data.get('phone', '')}")
        st.markdown(f"**National ID:** {st.session_state.form_data.get('nid', '')}")
        st.markdown(f"**Gender:** {st.session_state.form_data.get('gender', '')}")
    
    st.markdown("### Academic Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### SSC Details")
        st.markdown(f"**GPA:** {st.session_state.form_data.get('ssc_gpa', '')}")
        st.markdown(f"**Year:** {st.session_state.form_data.get('ssc_year', '')}")
        st.markdown(f"**Board:** {st.session_state.form_data.get('ssc_board', '')}")
        st.markdown(f"**Group:** {st.session_state.form_data.get('ssc_group', '')}")
    
    with col2:
        st.markdown("#### HSC Details")
        st.markdown(f"**GPA:** {st.session_state.form_data.get('hsc_gpa', '')}")
        st.markdown(f"**Year:** {st.session_state.form_data.get('hsc_year', '')}")
        st.markdown(f"**Board:** {st.session_state.form_data.get('hsc_board', '')}")
        st.markdown(f"**Group:** {st.session_state.form_data.get('hsc_group', '')}")
    
    st.markdown("### Program Selection")
    st.markdown(f"**Program:** {st.session_state.form_data.get('program_choice', '')}")
    st.markdown(f"**Semester:** {st.session_state.form_data.get('semester', '')} {st.session_state.form_data.get('year', '')}")
    
    st.markdown("### Documents")
    st.markdown(f"**Photo:** {st.session_state.form_data.get('photo', 'Not uploaded')}")
    st.markdown(f"**SSC Certificate:** {st.session_state.form_data.get('ssc_cert', 'Not uploaded')}")
    st.markdown(f"**HSC Certificate:** {st.session_state.form_data.get('hsc_cert', 'Not uploaded')}")
    if st.session_state.form_data.get('nid_copy'):
        st.markdown(f"**NID Copy:** {st.session_state.form_data.get('nid_copy', '')}")
    
    st.markdown("---")
    agree = st.checkbox("I certify that all information provided is true and accurate to the best of my knowledge.", 
                       value=st.session_state.form_data.get("agree", False), key="agree")
    st.session_state.form_data["agree"] = agree

def application_form():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    load_application_data()
    
    total_applications = len(st.session_state.submitted_applications)
    approved_applications = len([app for app in st.session_state.submitted_applications if app.get("status") == "Approved"])
    pending_applications = len([app for app in st.session_state.submitted_applications if app.get("status") == "Pending"])
    
    st.markdown("### Application Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_applications}</div><div class="stat-label">Total Applications</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{approved_applications}</div><div class="stat-label">Approved</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{pending_applications}</div><div class="stat-label">Pending</div></div>', unsafe_allow_html=True)
    
    progress = (st.session_state.current_step - 1) / 5 * 100
    st.markdown(f'<div class="progress-bar"><div class="progress-bar-fill" style="width: {progress}%"></div></div>', unsafe_allow_html=True)
    st.markdown(f'**Application Progress:** {progress:.0f}% Complete')
    
    steps = ["Personal Info", "Academic Info", "Program Selection", "Documents", "Review & Submit"]
    step_html = '<div class="step-container">'
    for i, step in enumerate(steps, 1):
        if i == st.session_state.current_step:
            step_html += f'<div class="step active">{step}</div>'
        elif i < st.session_state.current_step:
            step_html += f'<div class="step completed">{step}</div>'
        else:
            step_html += f'<div class="step">{step}</div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)
    
    if st.session_state.current_step == 1:
        step1_personal_info()
    elif st.session_state.current_step == 2:
        step2_academic_info()
    elif st.session_state.current_step == 3:
        step3_program_selection()
    elif st.session_state.current_step == 4:
        step4_documents()
    elif st.session_state.current_step == 5:
        step5_review()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_step > 1:
            if st.button("← Previous"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 5:
            if st.button("Next →"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("Submit Application", type="primary"):
                save_application_data(st.session_state.form_data.copy())
            
                st.markdown('<div class="success-animation">🎉 Application submitted successfully!</div>', unsafe_allow_html=True)

                st.session_state.form_data = {}
                st.session_state.current_step = 1
                    
                time.sleep(2)
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.submitted_applications:
        st.markdown("---")
        st.markdown("### Export Applications")
        excel_data = export_applications()
        if excel_data:
            st.download_button(
                label="Download Applications as Excel",
                data=excel_data,
                file_name="university_applications.xlsx",
                mime="application/vnd.ms-excel"
            )

# ============================
# 9. Dashboard
# ============================
def show_dashboard():
    st.markdown('<div class="main-header">Admission Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="social-proof">Trusted by 10,000+ Students Worldwide</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("Programs", len(programs)),
        ("Departments", len(departments)),
        ("Waivers", len(waivers)),
        ("Applications", 1250)
    ]
    for i, (label, value) in enumerate(stats):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stats-number">{value}</div>
                <div class="stats-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">Applications by Program</div>', unsafe_allow_html=True)
        program_names = [p["name"] for p in programs]
        num_programs = len(program_names)
        applications_data = [i * 100 // num_programs + 100 for i in range(num_programs)]
        program_data = pd.DataFrame({
            "Program": program_names,
            "Applications": applications_data
        })
        if program_data.empty:
            st.warning("No program data available to display.")
        else:
            fig = px.bar(program_data, x="Program", y="Applications", color="Program", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">Waiver Distribution</div>', unsafe_allow_html=True)
        waiver_names = [w["name"] for w in waivers]
        num_waivers = len(waiver_names)
        count_data = [i * 50 // num_waivers + 50 for i in range(num_waivers)]
        waiver_data = pd.DataFrame({
            "Waiver": waiver_names,
            "Count": count_data
        })
        if waiver_data.empty:
            st.warning("No waiver data available to display.")
        else:
            fig = px.pie(waiver_data, values="Count", names="Waiver", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ============================
# 10. Enhanced Waiver Calculator UI
# ============================
def render_waiver_section():
    st.markdown('<div class="main-header">Tuition Waiver Calculator</div>', unsafe_allow_html=True)
    
    student_type = st.radio("Are you a:", ["New Applicant", "Current Student"], horizontal=True)
    is_new_student = student_type == "New Applicant"
    
    with st.form("waiver_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            faculty = st.selectbox("Faculty*", 
                                  ["SIT_BE_AHS_Engineering", "Humanities_Social_Sciences", "BPharm_LLB_CSE"])
        with col2:
            hsc_gpa = st.number_input("HSC GPA*", 0.0, 5.0, 0.0, format="%.2f", step=0.01)
        
        ssc_gpa = st.number_input("SSC GPA*", 0.0, 5.0, 0.0, format="%.2f", step=0.01)
        
        if not is_new_student:
            current_sgpa = st.number_input("Current SGPA*", 0.0, 4.0, 0.0, format="%.2f", step=0.01)
        else:
            current_sgpa = 0
        
        st.markdown("### Student Profile")
        profile_col1, profile_col2 = st.columns(2)
        
        with profile_col1:
            is_female = st.checkbox("Female Student")
            is_diu_employee = st.checkbox("DIU Employee")
            is_dic_student = st.checkbox("DIC/Eminence/Human Development College Student")
            is_dipti_student = st.checkbox("DIPTI BM College Student")
            if is_dipti_student:
                hsc_better_than_ssc = st.checkbox("HSC result better than SSC")
            is_alumni_relative = st.checkbox("1st Blood Relative of DIU Alumni")
            
        with profile_col2:
            is_physically_challenged = st.checkbox("Physically Challenged")
            is_tribal = st.checkbox("Tribal/Ethnic Group Student")
            has_sibling_student = st.checkbox("Has Sibling Studying at DIU")
            has_spouse_student = st.checkbox("Has Spouse Studying at DIU")
            is_diploma_holder = st.checkbox("Diploma Holder")
            if is_diploma_holder:
                diploma_gpa = st.number_input("Diploma GPA", 0.0, 4.0, 0.0, format="%.2f", step=0.01)
            is_first_batch = st.checkbox("First Batch of Program")
            
        player_level = st.selectbox("Sports Achievement Level", 
                                   ["None", "National", "Premier League", "First Division", 
                                    "Second Division", "DIU Player"])
        
        submitted = st.form_submit_button("Calculate Eligible Waivers", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted and all([hsc_gpa, ssc_gpa]):
            student_profile = {
                "is_female": is_female,
                "is_diu_employee": is_diu_employee,
                "is_dic_student": is_dic_student,
                "is_dipti_student": is_dipti_student,
                "hsc_better_than_ssc": hsc_better_than_ssc if is_dipti_student else False,
                "is_alumni_relative": is_alumni_relative,
                "is_physically_challenged": is_physically_challenged,
                "is_tribal": is_tribal,
                "has_sibling_student": has_sibling_student,
                "has_spouse_student": has_spouse_student,
                "is_diploma_holder": is_diploma_holder,
                "diploma_gpa": diploma_gpa if is_diploma_holder else 0,
                "is_first_batch": is_first_batch,
                "player_level": player_level if player_level != "None" else None
            }
            
            eligible_waivers = calculate_waivers(
                hsc_gpa, ssc_gpa, faculty, is_new_student, current_sgpa, student_profile
            )
            if eligible_waivers:
                st.success(f"🎉 You are eligible for {len(eligible_waivers)} waiver(s)!")
                
                for waiver in eligible_waivers:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3>{waiver['type']} - {waiver['waiver_percentage']}% Waiver</h3>
                        <p><b>Condition:</b> {waiver['condition']}</p>
                        <p><b>Requirements:</b> {waiver['requirements']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                max_waiver = max([w['waiver_percentage'] for w in eligible_waivers], default=0)
                st.markdown(f"""
                <div class="glass-card" style="background: rgba(16, 185, 129, 0.1);">
                    <h3>Maximum Waiver: {max_waiver}%</h3>
                    <p>Note: If eligible for multiple waivers, you will receive the highest percentage waiver.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No waivers available based on your profile. Consider improving your SGPA or exploring other quota options.")

# ============================
# 11. Streamlit UI
# ============================

if "user_prefs" not in st.session_state:
    st.session_state.user_prefs = {"theme": "light"}
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

LOGO_URL = "https://i.pinimg.com/originals/72/2c/57/722c5710002814308caff02ffb294c0d.png"

st.markdown(f"""
<div class="sticky-header">
    <img src="{LOGO_URL}" width="80" alt="DIU Logo">
    <div>
        <button class="diu-button" onclick="window.location.href='#application'">Apply Now</button>
        <button class="diu-button" onclick="window.location.href='#chat'">Chat with Us</button>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">
        <img src="{LOGO_URL}" width="100" alt="DIU Logo">
        <h2 style="color: var(--text-primary);">DIU Premium Portal</h2>
    </div>
    """, unsafe_allow_html=True)
    
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.user_prefs["theme"] == "light" else 1)
    st.session_state.user_prefs["theme"] = theme.lower()
    
    nav_options = {
        "💬 Chat": "chat",
        "📊 Dashboard": "dashboard",
        "🎓 Recommendation": "recommendation",
        "📄 Application": "application",
        "💰 Waiver Calculator": "waiver",
        "ℹ️ Help": "help"
    }
    
    selected_nav = st.radio("Navigation", list(nav_options.keys()), label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown('<div class="social-proof">Trusted by 10,000+ Students</div>', unsafe_allow_html=True)
    if st.button("Save Preferences"):
        st.success("Preferences saved!")

if nav_options[selected_nav] == "dashboard":
    show_dashboard()

elif nav_options[selected_nav] == "chat":
    st.markdown('<div class="main-header">Chat with DIU Assistant</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f'<div class={"user-message" if speaker == "user" else "bot-message"}><b>{"You" if speaker == "user" else "DIU Assistant"}:</b> {msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Ask about admissions...", 
                key="chat_input_key",
                label_visibility="collapsed",
                value=st.session_state.chat_input
            )
        with col2:
            send_button = st.form_submit_button("Send", type="primary")
        
        if send_button and user_input:
            st.session_state.chat_history.append(("user", user_input))
            
            typing_placeholder = st.empty()
            typing_placeholder.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            
            time.sleep(1)
            response = get_bot_response(user_input)
            
            st.session_state.chat_history.append(("bot", response))
            
            typing_placeholder.empty()
            
            st.session_state.chat_input = ""
            
            st.rerun()

elif nav_options[selected_nav] == "recommendation":
    st.markdown('<div class="main-header">Department Recommendation</div>', unsafe_allow_html=True)
    with st.form("recommendation_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gpa_ssc = st.number_input("SSC GPA*", 0.0, 5.0, 0.0, format="%.2f")
        with col2:
            gpa_hsc = st.number_input("HSC GPA*", 0.0, 5.0, 0.0, format="%.2f")
        interests = st.text_area("Interests & Skills*", placeholder="e.g., programming, business")
        submitted = st.form_submit_button("Get Recommendation", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted and all([gpa_ssc, gpa_hsc, interests]):
            recommended = recommend_department(interests, gpa_hsc, gpa_ssc)
            if recommended:
                for dept in recommended:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3>{dept['name']}</h3>
                        <p><b>Match Score:</b> {dept['score']}/5</p>
                        <p>{dept['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)

elif nav_options[selected_nav] == "application":
    st.markdown('<div class="main-header">Admission Application</div>', unsafe_allow_html=True)
    application_form()

elif nav_options[selected_nav] == "waiver":
    render_waiver_section()

elif nav_options[selected_nav] == "help":
    st.markdown('<div class="main-header">Help & Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <h3>Frequently Asked Questions</h3>
    <p>Explore common questions about admissions and programs.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>© 2025 Daffodil International University</p>
    <p>
        <a href="https://daffodilvarsity.edu.bd/" style="color: var(--secondary);">DIU Website</a> |
        <a href="https://daffodilvarsity.edu.bd/contact" style="color: var(--secondary);">Contact Us</a>
    </p>
    <p>Follow us: <a href="#">Facebook</a> | <a href="#">Twitter</a> | <a href="#">LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)

