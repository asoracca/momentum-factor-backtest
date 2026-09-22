"""SQLite schema v1 and parameterized research queries."""

import json
import sqlite3

import pandas as pd

SCHEMA_VERSION = 1


def connect(path):
    db = sqlite3.connect(path)
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, SCHEMA_VERSION):
        db.close()
        raise ValueError(f"unsupported database schema {version}")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS experiments(run_id TEXT PRIMARY KEY, manifest TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS eligibility(
      run_id TEXT REFERENCES experiments, date TEXT, ticker TEXT, member INTEGER,
      observed INTEGER, eligible INTEGER, signal REAL,
      PRIMARY KEY(run_id,date,ticker));
    CREATE TABLE IF NOT EXISTS asset_returns(
      run_id TEXT REFERENCES experiments, date TEXT, ticker TEXT, value REAL,
      PRIMARY KEY(run_id,date,ticker));
    CREATE TABLE IF NOT EXISTS holdings(
      run_id TEXT REFERENCES experiments, mode TEXT, formation_date TEXT,
      return_date TEXT, ticker TEXT, weight REAL,
      PRIMARY KEY(run_id,mode,formation_date,ticker));
    CREATE TABLE IF NOT EXISTS monthly_returns(
      run_id TEXT REFERENCES experiments, mode TEXT, date TEXT, gross REAL,
      turnover REAL, missing_held INTEGER,
      PRIMARY KEY(run_id,mode,date));
    CREATE TABLE IF NOT EXISTS costs(
      run_id TEXT, mode TEXT, date TEXT, bps REAL, cost REAL, net REAL,
      PRIMARY KEY(run_id,mode,date,bps),
      FOREIGN KEY(run_id,mode,date) REFERENCES monthly_returns(run_id,mode,date));
    CREATE TABLE IF NOT EXISTS results(
      run_id TEXT REFERENCES experiments, mode TEXT, bps REAL, block INTEGER,
      result TEXT NOT NULL, PRIMARY KEY(run_id,mode,bps,block));
    PRAGMA user_version=1;
    """)
    return db


def number(value):
    return None if pd.isna(value) else float(value)


def save_run(db, manifest, portfolios, summary):
    run = manifest["run_id"]
    with db:
        db.execute(
            "INSERT INTO experiments VALUES (?,?)",
            (run, json.dumps(manifest, sort_keys=True)),
        )
        prices, member, eligible, scores, _, _, _, returns = next(
            iter(portfolios.values())
        )
        db.executemany(
            "INSERT INTO eligibility VALUES (?,?,?,?,?,?,?)",
            [
                (
                    run,
                    date.strftime("%Y-%m-%d"),
                    ticker,
                    int(member.loc[date, ticker]),
                    int(pd.notna(prices.loc[date, ticker])),
                    int(eligible.loc[date, ticker]),
                    number(scores.loc[date, ticker]),
                )
                for date in prices.index
                for ticker in prices.columns
            ],
        )
        db.executemany(
            "INSERT INTO asset_returns VALUES (?,?,?,?)",
            [
                (
                    run,
                    date.strftime("%Y-%m-%d"),
                    ticker,
                    number(returns.loc[date, ticker]),
                )
                for date in prices.index
                for ticker in prices.columns
            ],
        )
        for mode, portfolio in portfolios.items():
            _, _, _, _, weights, details, missing, _ = portfolio
            db.executemany(
                "INSERT INTO holdings VALUES (?,?,?,?,?,?)",
                [
                    (
                        run,
                        mode,
                        weights.index[i].strftime("%Y-%m-%d"),
                        weights.index[i + 1].strftime("%Y-%m-%d"),
                        ticker,
                        float(weights.iloc[i][ticker]),
                    )
                    for i in range(12, len(weights) - 1)
                    for ticker in weights.columns
                ],
            )
            for date, row in details.iterrows():
                day = date.strftime("%Y-%m-%d")
                db.execute(
                    "INSERT INTO monthly_returns VALUES (?,?,?,?,?,?)",
                    (
                        run,
                        mode,
                        day,
                        number(row.gross_return),
                        row.turnover,
                        int(missing.loc[date]),
                    ),
                )
                for bps in manifest["config"]["costs"]:
                    cost = row.turnover * bps / 10000
                    db.execute(
                        "INSERT INTO costs VALUES (?,?,?,?,?,?)",
                        (run, mode, day, bps, cost, number(row.gross_return - cost)),
                    )
        for row in summary.to_dict("records"):
            db.execute(
                "INSERT INTO results VALUES (?,?,?,?,?)",
                (
                    run,
                    row["mode"],
                    row["bps"],
                    row["block"],
                    json.dumps(
                        {
                            k: None if isinstance(v, float) and pd.isna(v) else v
                            for k, v in row.items()
                        },
                        allow_nan=False,
                    ),
                ),
            )


def comparison(db, run_id, start, bps):
    """Keep incomplete comparisons visibly unknown, rather than averaging gaps."""
    return pd.read_sql_query(
        """
    SELECT r.mode, COUNT(*) AS months, COUNT(c.net) AS observed_months,
      CASE WHEN COUNT(c.net)=COUNT(*) THEN AVG(c.net)*12 END AS annual_mean,
      AVG(r.turnover) AS mean_turnover, SUM(c.cost) AS total_cost
    FROM monthly_returns r JOIN costs c USING(run_id,mode,date)
    WHERE r.run_id=? AND r.date>=? AND c.bps=? GROUP BY r.mode ORDER BY r.mode
    """,
        db,
        params=(run_id, start, bps),
    )


def coverage(db, run_id):
    return pd.read_sql_query(
        """SELECT ticker, COUNT(*) AS months,
      SUM(1-observed) AS missing_prices, SUM(member) AS member_months,
      SUM(eligible) AS eligible_months FROM eligibility
      WHERE run_id=? GROUP BY ticker ORDER BY ticker""",
        db,
        params=(run_id,),
    )


def attribution(db, run_id, mode, bps):
    """One row per held asset; portfolio cost repeats and must not be summed."""
    return pd.read_sql_query(
        """
      SELECT h.formation_date,h.return_date,h.ticker,e.eligible,h.weight,
        a.value AS asset_return,h.weight*a.value AS contribution,
        r.gross,r.missing_held,c.cost AS portfolio_cost,c.net AS portfolio_net
      FROM holdings h
      JOIN eligibility e ON e.run_id=h.run_id AND e.date=h.formation_date AND e.ticker=h.ticker
      JOIN asset_returns a ON a.run_id=h.run_id AND a.date=h.return_date AND a.ticker=h.ticker
      JOIN monthly_returns r ON r.run_id=h.run_id AND r.mode=h.mode AND r.date=h.return_date
      JOIN costs c ON c.run_id=r.run_id AND c.mode=r.mode AND c.date=r.date
      WHERE h.run_id=? AND h.mode=? AND c.bps=? AND h.weight<>0
      ORDER BY h.return_date,h.ticker
    """,
        db,
        params=(run_id, mode, bps),
    )
