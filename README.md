# GaiaChat

**Natural Language Interface for Gaia DR3 Data**

Built for the [MIT Gaia DR3 Hackathon](https://gaiadr3hack.mit.edu) (January 29-31, 2026)

---

## Demo

<video src="https://github.com/adamzacharia/Gaia/raw/master/Recording%202026-02-08%20165501.mp4" controls="controls" style="max-width: 100%;">
</video>

*GaiaChat features a modern interface with feature cards, smart visualizations, and natural language querying of the Gaia stellar catalog.*

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Language | Python 3.11 |
| UI Framework | Streamlit |
| Data Access | astroquery (Gaia TAP/ADQL) |
| Astronomy | astropy |
| Visualization | matplotlib, plotly |
| Environment | Conda |

---

## What is GaiaChat?

GaiaChat lets you explore the Gaia DR3 stellar catalog using natural language. No ADQL knowledge required!

**Example queries:**
- "Show me the nearest 100 stars"
- "Search for stars around RA=180, Dec=45 with a 2 degree radius"
- "Find stars within 3 kpc with total space velocities exceeding 350 km/s"
- "Find bright stars with parallax greater than 10 mas"

## Features

- **Natural Language to ADQL**: Automatically translates your questions into proper queries
- **Smart Visualizations**: HR diagrams, sky maps, velocity plots, Toomre diagrams
- **Dark Matter Ready**: Built-in queries for stellar streams, accreted stars, and kinematics
- **Science Explanations**: Get context about your results

## Quick Start

```bash
# Activate environment
conda activate gaia

# Navigate to project
cd Gaia

# Run the app
streamlit run ui/app.py
```

## Project Structure

```
Gaia/
├── core/
│   ├── config.py       # Configuration management
│   ├── gaia_service.py # Gaia TAP/ADQL interface
│   └── agent.py        # LLM agent with tools
├── visualization/
│   └── plots.py        # Astronomical visualizations
├── ui/
│   └── app.py          # Streamlit interface
├── .env                # API keys (not in git)
└── requirements.txt    # Dependencies
```

## Science Background

This tool is designed with dark matter research in mind, supporting queries relevant to:
- **Stellar streams** (Nyx, Gaia Sausage/Enceladus)
- **Accreted vs in-situ star classification**
- **Kinematic substructure** in the solar neighborhood
- **Phase-space analysis** for dark matter velocity distributions

## Credits

Made by **Adam Zacharia Anil** | Contact: adamanil@mit.edu

GaiaChat translates natural language into ADQL queries for the Gaia DR3 stellar catalog.
