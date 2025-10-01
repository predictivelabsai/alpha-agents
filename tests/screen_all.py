from .fundamental_agent import FundamentalAgent
from utils.data import INDUSTRIES
from tqdm import tqdm


def main():
    for industry in tqdm(INDUSTRIES):
        agent = FundamentalAgent()
        df = agent.screen_sector("US", industries=industry, verbose=False)
        df.to_excel(f"test-data/screener/screening_results_20250919/{industry}.xlsx")


if __name__ == "__main__":
    main()
