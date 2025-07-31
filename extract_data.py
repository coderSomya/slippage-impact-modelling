#!/usr/bin/env python3
"""
Data Extraction Script for Market Impact Research

This script extracts the research_data.zip file into the Data folder.
The Data folder contains high-frequency limit order book data for FROG, CRWV, and SOUN stocks.
"""

import zipfile
import os
import sys
from pathlib import Path

def extract_research_data():
    """Extract research_data.zip into Data folder"""
    
    # Define paths
    zip_file = "research_data.zip"
    extract_dir = "Data"
    
    print("Market Impact Research - Data Extraction")
    print("=" * 50)
    
    # Check if zip file exists
    if not os.path.exists(zip_file):
        print(f"❌ Error: {zip_file} not found!")
        print("Please ensure research_data.zip is in the current directory.")
        return False
    
    # Create Data directory if it doesn't exist
    if not os.path.exists(extract_dir):
        print(f"Creating {extract_dir} directory...")
        os.makedirs(extract_dir)
    
    try:
        print(f"📦 Extracting {zip_file} to {extract_dir}/...")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total number of files for progress tracking
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"Found {total_files} files to extract...")
            
            # Extract all files
            zip_ref.extractall(extract_dir)
            
            print("✅ Extraction completed successfully!")
            
            # List extracted files
            extracted_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.csv'):
                        extracted_files.append(os.path.join(root, file))
            
            print(f"\n📊 Extracted {len(extracted_files)} CSV files:")
            for file in sorted(extracted_files):
                print(f"  - {file}")
            
            # Check for expected stock files
            expected_stocks = ['FROG', 'CRWV', 'SOUN']
            found_stocks = set()
            
            for file in extracted_files:
                for stock in expected_stocks:
                    if stock in file:
                        found_stocks.add(stock)
            
            print(f"\n🎯 Stock data found:")
            for stock in expected_stocks:
                if stock in found_stocks:
                    print(f"  ✅ {stock}")
                else:
                    print(f"  ❌ {stock} (not found)")
            
            if len(found_stocks) == len(expected_stocks):
                print("\n🎉 All expected stock data extracted successfully!")
                return True
            else:
                print(f"\n⚠️  Warning: Only {len(found_stocks)}/{len(expected_stocks)} stocks found")
                return False
                
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_file} is not a valid ZIP file!")
        return False
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return False

def main():
    """Main function"""
    
    print("Starting data extraction...")
    
    success = extract_research_data()
    
    if success:
        print("\n✅ Data extraction completed successfully!")
        print("You can now run the research scripts:")
        print("  python test_data_loading.py")
        print("  python comprehensive_eda.py")
        print("  python market_impact_analysis.py")
        print("  python market_impact_research.py")
        print("  python academic_market_impact_research.py")
    else:
        print("\n❌ Data extraction failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 