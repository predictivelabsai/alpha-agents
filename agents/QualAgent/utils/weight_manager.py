#!/usr/bin/env python3
"""
Interactive Weight Management System for QualAgent
Allows users to view, modify, and approve scoring weights for expert-guided analysis
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from engines.enhanced_scoring_system import WeightingScheme

class InteractiveWeightManager:
    """Interactive weight management for expert analysis"""

    def __init__(self):
        self.current_weights = WeightingScheme()
        self.weight_descriptions = {
            'barriers_to_entry': 'Capital, regulatory, and technical barriers preventing new competitors',
            'brand_monopoly': 'Brand strength, customer loyalty, and pricing power advantages',
            'economies_of_scale': 'Cost advantages from size, volume discounts, operational leverage',
            'network_effects': 'Value increases with more users/participants in the platform',
            'switching_costs': 'Customer difficulty and cost to switch to competitor products',
            'competitive_differentiation': 'Unique value propositions and sustainable advantages',
            'technology_moats': 'Technical barriers, IP strength, R&D leadership',
            'market_timing': 'Strategic positioning for market trends and adoption curves',
            'management_quality': 'Leadership effectiveness, execution capability, strategic vision',
            'key_growth_drivers': 'Revenue catalysts, expansion opportunities, market trends',
            'transformation_potential': 'Ability to adapt, enter new markets, evolve business model',
            'platform_expansion': 'Ecosystem growth, adjacent market entry, platform scaling',
            'major_risk_factors': 'Business threats, competitive risks, market headwinds (NEGATIVE)',
            'red_flags': 'Warning signals, concerning trends, governance issues (NEGATIVE)'
        }

    def show_current_weights(self):
        """Display current weight configuration"""
        print("\n" + "="*70)
        print("📊 CURRENT WEIGHT CONFIGURATION")
        print("="*70)

        # Group weights by category
        competitive_moats = {
            'barriers_to_entry': self.current_weights.barriers_to_entry,
            'brand_monopoly': self.current_weights.brand_monopoly,
            'economies_of_scale': self.current_weights.economies_of_scale,
            'network_effects': self.current_weights.network_effects,
            'switching_costs': self.current_weights.switching_costs
        }

        strategic_insights = {
            'competitive_differentiation': self.current_weights.competitive_differentiation,
            'technology_moats': self.current_weights.technology_moats,
            'market_timing': self.current_weights.market_timing,
            'management_quality': self.current_weights.management_quality,
            'key_growth_drivers': self.current_weights.key_growth_drivers,
            'transformation_potential': self.current_weights.transformation_potential,
            'platform_expansion': self.current_weights.platform_expansion
        }

        risk_factors = {
            'major_risk_factors': self.current_weights.major_risk_factors,
            'red_flags': self.current_weights.red_flags
        }

        # Display by category
        self._display_category("🏰 COMPETITIVE MOATS", competitive_moats)
        self._display_category("📈 STRATEGIC INSIGHTS", strategic_insights)
        self._display_category("⚠️  RISK FACTORS", risk_factors)

        # Calculate totals
        positive_total = sum(competitive_moats.values()) + sum(strategic_insights.values())
        negative_total = sum(risk_factors.values())
        net_total = positive_total + negative_total

        print(f"\n📋 WEIGHT SUMMARY:")
        print(f"  • Positive Weights: {positive_total:.3f}")
        print(f"  • Negative Weights: {negative_total:.3f}")
        print(f"  • Net Total: {net_total:.3f}")

        if abs(net_total - 1.0) > 0.001:
            print(f"  ⚠️  WARNING: Weights don't sum to 1.0 (will be normalized)")

    def _display_category(self, title: str, weights: Dict[str, float]):
        """Display weights for a category"""
        print(f"\n{title}")
        print("-" * 50)
        total = sum(weights.values())
        for name, weight in weights.items():
            percentage = (weight / total * 100) if total != 0 else 0
            description = self.weight_descriptions.get(name, "No description")
            print(f"  {name.replace('_', ' ').title():<25} {weight:>6.3f} ({percentage:4.1f}%)")
            print(f"    └─ {description}")

    def interactive_weight_approval(self) -> WeightingScheme:
        """Interactive weight approval process"""
        print("\n" + "="*70)
        print("🎯 INTERACTIVE WEIGHT APPROVAL")
        print("="*70)

        while True:
            self.show_current_weights()

            print(f"\n🔧 WEIGHT MANAGEMENT OPTIONS:")
            print("1. ✅ Approve current weights")
            print("2. 🎨 Apply investment philosophy preset")
            print("3. ✏️  Modify specific weights")
            print("4. 💾 Save weight configuration")
            print("5. 📂 Load weight configuration")
            print("6. 🔄 Reset to defaults")

            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                print("\n✅ Weights approved!")
                return self.current_weights.normalize_weights()

            elif choice == "2":
                self._apply_philosophy_preset()

            elif choice == "3":
                self._modify_specific_weights()

            elif choice == "4":
                self._save_weight_configuration()

            elif choice == "5":
                self._load_weight_configuration()

            elif choice == "6":
                self.current_weights = WeightingScheme()
                print("\n🔄 Reset to default weights")

            else:
                print("❌ Invalid choice. Please select 1-6.")

    def _apply_philosophy_preset(self):
        """Apply investment philosophy presets"""
        print(f"\n🎨 INVESTMENT PHILOSOPHY PRESETS:")
        print("1. 📈 Growth Focus - Emphasize growth drivers and transformation")
        print("2. 🏰 Value Focus - Emphasize competitive moats and barriers")
        print("3. 💎 Quality Focus - Emphasize management and differentiation")
        print("4. 🛡️  Risk-Aware - Emphasize risk assessment and stability")
        print("5. ⚡ Tech Focus - Emphasize technology moats and innovation")

        philosophy = input("\nSelect philosophy (1-5): ").strip()

        if philosophy == "1":  # Growth Focus
            self.current_weights.key_growth_drivers = 0.25
            self.current_weights.transformation_potential = 0.20
            self.current_weights.platform_expansion = 0.15
            self.current_weights.market_timing = 0.12
            self.current_weights.barriers_to_entry = 0.10
            print("📈 Applied Growth Focus weights")

        elif philosophy == "2":  # Value Focus
            self.current_weights.barriers_to_entry = 0.30
            self.current_weights.brand_monopoly = 0.25
            self.current_weights.economies_of_scale = 0.20
            self.current_weights.switching_costs = 0.15
            self.current_weights.major_risk_factors = -0.10
            print("🏰 Applied Value Focus weights")

        elif philosophy == "3":  # Quality Focus
            self.current_weights.management_quality = 0.25
            self.current_weights.competitive_differentiation = 0.20
            self.current_weights.brand_monopoly = 0.18
            self.current_weights.barriers_to_entry = 0.15
            self.current_weights.red_flags = -0.08
            print("💎 Applied Quality Focus weights")

        elif philosophy == "4":  # Risk-Aware
            self.current_weights.major_risk_factors = -0.15
            self.current_weights.red_flags = -0.10
            self.current_weights.barriers_to_entry = 0.25
            self.current_weights.switching_costs = 0.20
            self.current_weights.management_quality = 0.15
            print("🛡️  Applied Risk-Aware weights")

        elif philosophy == "5":  # Tech Focus
            self.current_weights.technology_moats = 0.30
            self.current_weights.network_effects = 0.20
            self.current_weights.platform_expansion = 0.15
            self.current_weights.transformation_potential = 0.15
            self.current_weights.competitive_differentiation = 0.12
            print("⚡ Applied Tech Focus weights")

        else:
            print("❌ Invalid choice")

    def _modify_specific_weights(self):
        """Modify specific weight values"""
        print(f"\n✏️  MODIFY SPECIFIC WEIGHTS")
        print("Enter new weight values (0.0-1.0) or press Enter to keep current value")
        print("Note: Negative weights are allowed for risk factors\n")

        weight_attrs = [
            'barriers_to_entry', 'brand_monopoly', 'economies_of_scale',
            'network_effects', 'switching_costs', 'competitive_differentiation',
            'technology_moats', 'market_timing', 'management_quality',
            'key_growth_drivers', 'transformation_potential', 'platform_expansion',
            'major_risk_factors', 'red_flags'
        ]

        for attr in weight_attrs:
            current_value = getattr(self.current_weights, attr)
            display_name = attr.replace('_', ' ').title()
            description = self.weight_descriptions.get(attr, "")

            print(f"\n{display_name}")
            print(f"  Current: {current_value:.3f}")
            print(f"  Description: {description}")

            user_input = input(f"  New value [{current_value:.3f}]: ").strip()

            if user_input:
                try:
                    new_value = float(user_input)
                    if attr in ['major_risk_factors', 'red_flags']:
                        # Allow negative values for risk factors
                        if -1.0 <= new_value <= 0.0:
                            setattr(self.current_weights, attr, new_value)
                            print(f"  ✅ Updated to {new_value:.3f}")
                        else:
                            print(f"  ❌ Risk factors must be between -1.0 and 0.0")
                    else:
                        # Positive weights only for other factors
                        if 0.0 <= new_value <= 1.0:
                            setattr(self.current_weights, attr, new_value)
                            print(f"  ✅ Updated to {new_value:.3f}")
                        else:
                            print(f"  ❌ Value must be between 0.0 and 1.0")
                except ValueError:
                    print(f"  ❌ Invalid input '{user_input}', keeping current value")

    def _save_weight_configuration(self):
        """Save current weight configuration"""
        filename = input("\nEnter filename (without .json): ").strip()
        if not filename:
            filename = f"weight_config_{int(time.time())}"

        filepath = f"{filename}.json"

        weight_data = {
            'timestamp': time.time(),
            'description': f"Custom weight configuration saved at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            'weights': {
                attr: getattr(self.current_weights, attr)
                for attr in dir(self.current_weights)
                if not attr.startswith('_') and hasattr(self.current_weights, attr)
                and isinstance(getattr(self.current_weights, attr), (int, float))
            }
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(weight_data, f, indent=2)
            print(f"💾 Weights saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving weights: {e}")

    def _load_weight_configuration(self):
        """Load weight configuration from file"""
        filename = input("\nEnter filename (with .json extension): ").strip()

        try:
            with open(filename, 'r') as f:
                weight_data = json.load(f)

            if 'weights' in weight_data:
                for attr, value in weight_data['weights'].items():
                    if hasattr(self.current_weights, attr):
                        setattr(self.current_weights, attr, value)

                print(f"📂 Weights loaded from {filename}")
                if 'description' in weight_data:
                    print(f"   Description: {weight_data['description']}")
            else:
                print("❌ Invalid weight file format")

        except FileNotFoundError:
            print(f"❌ File {filename} not found")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

def main():
    """Main interactive weight management"""
    print("🎯 QUALAGENT WEIGHT MANAGEMENT SYSTEM")
    print("="*70)

    manager = InteractiveWeightManager()

    print("This tool allows you to:")
    print("• View current weight configuration")
    print("• Apply investment philosophy presets")
    print("• Modify individual weights")
    print("• Save/load weight configurations")

    approved_weights = manager.interactive_weight_approval()

    print(f"\n🎯 FINAL APPROVED WEIGHTS:")
    manager.current_weights = approved_weights
    manager.show_current_weights()

    # Save for use in analysis
    save_for_analysis = input(f"\n💾 Save these weights for next analysis? (y/N): ").lower() == 'y'
    if save_for_analysis:
        with open('approved_weights.json', 'w') as f:
            weight_data = {
                'timestamp': time.time(),
                'approved_by': input("Enter your name/ID: ").strip(),
                'weights': {
                    attr: getattr(approved_weights, attr)
                    for attr in dir(approved_weights)
                    if not attr.startswith('_') and hasattr(approved_weights, attr)
                    and isinstance(getattr(approved_weights, attr), (int, float))
                }
            }
            json.dump(weight_data, f, indent=2)
        print("💾 Weights saved to approved_weights.json")
        print("\nTo use these weights in analysis:")
        print("python run_enhanced_analysis.py --user-id expert1 --company MSFT --analysis-type expert_guided --custom-weights approved_weights.json")

if __name__ == "__main__":
    main()