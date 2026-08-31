"""Point-in-time earnings events and option-relevant reaction targets.

The I/B/E/S summary file contains many historical consensus snapshots for one
fiscal quarter.  ``download_earnings_events`` retains the final snapshot whose
statistical date is strictly before the actual announcement, then applies
date-valid I/B/E/S -> CRSP -> OptionMetrics links.  No post-announcement
estimate enters the feature table.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def wrds_connection(env_file: str | Path = ".env"):
    """Open WRDS without printing or persisting credentials."""
    import wrds
    from dotenv import load_dotenv

    load_dotenv(env_file)
    username = os.getenv("username")
    password = os.getenv("password")
    if not username or not password:
        raise RuntimeError("WRDS username/password are missing from .env")
    return wrds.Connection(wrds_username=username, wrds_password=password)


def earnings_event_sql(start: str, end: str) -> str:
    """SQL for one leakage-safe row per quarterly EPS announcement."""
    return f"""
        with forecasts as (
            select distinct on (ticker, fpedats, anndats_act)
                ticker, oftic, cname, fpedats, anndats_act, anntims_act,
                statpers, meanest, medest, stdev, numest, actual
            from ibes.statsum_epsus
            where measure = 'EPS'
              and fiscalp = 'QTR'
              and fpi = '6'
              and anndats_act between '{start}' and '{end}'
              and statpers < anndats_act
              and actual is not null
              and meanest is not null
            order by ticker, fpedats, anndats_act, statpers desc
        ), linked as (
            select distinct on (f.ticker, f.fpedats, f.anndats_act)
                f.*, i.permno, o.secid,
                i.score as ibes_link_score, o.score as option_link_score
            from forecasts f
            join wrdsapps_link_crsp_ibes.ibcrsphist i
              on i.ticker = f.ticker
             and f.anndats_act between i.sdate::date and i.edate::date
             and i.score <= 1
            join wrdsapps_link_crsp_optionm.opcrsphist o
              on o.permno = i.permno
             and f.anndats_act between o.sdate::date and o.edate::date
             and o.score <= 1
            order by f.ticker, f.fpedats, f.anndats_act, i.score, o.score
        )
        select * from linked
        order by anndats_act, anntims_act, ticker
    """


def _announcement_session(times: pd.Series) -> pd.Series:
    """Classify I/B/E/S Eastern announcement times conservatively."""
    text = times.astype("string")
    hour = pd.to_numeric(text.str.slice(0, 2), errors="coerce")
    minute = pd.to_numeric(text.str.slice(3, 5), errors="coerce")
    clock = hour * 60 + minute
    session = np.select(
        [clock < 9 * 60 + 30, clock >= 16 * 60,
         clock.notna()],
        ["before_open", "after_close", "market_hours"],
        default="unknown",
    )
    return pd.Series(session, index=times.index, dtype="string")


def prepare_earnings_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and create only pre-event consensus features."""
    out = frame.copy()
    for column in ("fpedats", "anndats_act", "statpers"):
        out[column] = pd.to_datetime(out[column])
    out["announcement_session"] = _announcement_session(out["anntims_act"])
    out["forecast_age_days"] = (
        out["anndats_act"] - out["statpers"]
    ).dt.days.astype("int16")
    scale = np.maximum(out["stdev"].fillna(0).abs().to_numpy(), 0.01)
    surprise = out["actual"].to_numpy() - out["meanest"].to_numpy()
    out["eps_surprise"] = surprise
    out["eps_surprise_z"] = np.clip(surprise / scale, -20, 20)
    denominator = np.maximum(out["meanest"].abs().to_numpy(), 0.05)
    out["eps_surprise_pct"] = np.clip(surprise / denominator, -10, 10)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    if not (out["statpers"] < out["anndats_act"]).all():
        raise RuntimeError("post-announcement consensus leaked into earnings table")
    return out


def download_earnings_events(start: str, end: str, output: str | Path,
                             env_file: str | Path = ".env") -> pd.DataFrame:
    """Download and cache the linked quarterly earnings-event table."""
    connection = wrds_connection(env_file)
    try:
        frame = connection.raw_sql(earnings_event_sql(start, end))
    finally:
        connection.close()
    frame = prepare_earnings_events(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def download_crsp_industry_history(output: str | Path,
                                   env_file: str | Path = ".env") -> pd.DataFrame:
    """Cache date-valid CRSP SIC/NAICS classifications."""
    connection = wrds_connection(env_file)
    try:
        frame = connection.raw_sql("""
            select permno, namedt, nameendt, shrcd, exchcd, siccd, naics,
                   ticker as crsp_ticker, comnam
            from crsp.msenames
            where permno is not null
            order by permno, namedt
        """)
    finally:
        connection.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def option_entry_straddle_sql(year: int, max_days: int = 28) -> str:
    """Return one near-ATM call/put pair per event and expiration.

    The quote cutoff respects announcement timing: after-close releases may
    use that day's close, while before-open releases use the preceding trading
    day. Pairing and ATM ranking happen on WRDS so raw chains stay server-side.
    """
    events = earnings_event_sql(f"{year}-01-01", f"{year}-12-31").strip().rstrip(";")
    return f"""
        with linked as (
            {events}
        ), timed as (
            select l.*,
                   case
                     when substring(l.anntims_act::text, 1, 5) < '09:30'
                       then 'before_open'
                     when substring(l.anntims_act::text, 1, 5) >= '16:00'
                       then 'after_close'
                   end as announcement_session
            from linked l
        ), quoted as (
            select t.ticker, t.oftic, t.fpedats, t.anndats_act,
                   t.anntims_act, t.announcement_session, t.permno, t.secid,
                   q.date as entry_date, q.exdate, q.strike_price,
                   q.cp_flag, q.best_bid, q.best_offer, q.impl_volatility,
                   q.delta, q.open_interest, q.volume, q.optionid, u.close,
                   dense_rank() over (
                     partition by t.ticker, t.fpedats, t.anndats_act, t.secid
                     order by q.date desc
                   ) as quote_rank
            from timed t
            join optionm.opprcd{year} q
              on q.secid = t.secid
             and q.date >= t.anndats_act - interval '7 days'
             and q.date <= case
                   when t.announcement_session = 'after_close'
                     then t.anndats_act
                   else t.anndats_act - interval '1 day'
                 end
            join optionm.secprd{year} u
              on u.secid = q.secid and u.date = q.date
            where t.announcement_session is not null
              and q.exdate >= case
                    when t.announcement_session = 'before_open'
                      then t.anndats_act
                    else t.anndats_act + interval '1 day'
                  end
              and q.exdate <= t.anndats_act + interval '{max_days} days'
              and q.ss_flag = '0'
              and q.best_bid >= 0
              and q.best_offer >= q.best_bid
              and q.best_offer > 0
              and q.impl_volatility > 0
              and abs(q.delta) between 0.15 and 0.85
        ), entries as (
            select * from quoted where quote_rank = 1
        ), pairs as (
            select e.ticker, e.oftic, e.fpedats, e.anndats_act,
                   e.anntims_act, e.announcement_session, e.permno, e.secid,
                   e.entry_date, e.exdate, e.strike_price / 1000.0 as strike,
                   e.close as underlying_close,
                   max(e.best_bid) filter (where e.cp_flag = 'C') as call_bid,
                   max(e.best_offer) filter (where e.cp_flag = 'C') as call_ask,
                   max(e.impl_volatility) filter (where e.cp_flag = 'C') as call_iv,
                   max(e.delta) filter (where e.cp_flag = 'C') as call_delta,
                   max(e.open_interest) filter (where e.cp_flag = 'C') as call_oi,
                   max(e.volume) filter (where e.cp_flag = 'C') as call_volume,
                   max(e.optionid) filter (where e.cp_flag = 'C') as call_optionid,
                   max(e.best_bid) filter (where e.cp_flag = 'P') as put_bid,
                   max(e.best_offer) filter (where e.cp_flag = 'P') as put_ask,
                   max(e.impl_volatility) filter (where e.cp_flag = 'P') as put_iv,
                   max(e.delta) filter (where e.cp_flag = 'P') as put_delta,
                   max(e.open_interest) filter (where e.cp_flag = 'P') as put_oi,
                   max(e.volume) filter (where e.cp_flag = 'P') as put_volume,
                   max(e.optionid) filter (where e.cp_flag = 'P') as put_optionid
            from entries e
            group by e.ticker, e.oftic, e.fpedats, e.anndats_act,
                     e.anntims_act, e.announcement_session, e.permno, e.secid,
                     e.entry_date, e.exdate, e.strike_price, e.close
            having count(*) filter (where e.cp_flag = 'C') > 0
               and count(*) filter (where e.cp_flag = 'P') > 0
        ), ranked as (
            select p.*,
                   row_number() over (
                     partition by ticker, fpedats, anndats_act, secid, exdate
                     order by abs(strike - underlying_close),
                              (call_ask - call_bid) + (put_ask - put_bid), strike
                   ) as distance_rank
            from pairs p
        )
        select * from ranked where distance_rank <= 3
        order by anndats_act, secid, exdate
    """


def download_option_entry_straddles(
        start_year: int, end_year: int, output: str | Path,
        env_file: str | Path = ".env", max_days: int = 28) -> pd.DataFrame:
    """Download compact pre-announcement ATM straddles year by year."""
    connection = wrds_connection(env_file)
    pieces = []
    try:
        for year in range(start_year, end_year + 1):
            print(f"OptionMetrics entry straddles: {year}", flush=True)
            pieces.append(connection.raw_sql(option_entry_straddle_sql(year, max_days)))
    finally:
        connection.close()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    for column in ("fpedats", "anndats_act", "entry_date", "exdate"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def option_event_batch_sql(events: pd.DataFrame, year: int,
                           max_days: int = 28) -> str:
    """Compact raw-chain query for an explicit local batch of event keys."""
    rows = []
    for event in events.itertuples(index=False):
        announcement = pd.Timestamp(event.anndats_act).date().isoformat()
        cutoff = pd.Timestamp(event.option_quote_cutoff).date().isoformat()
        session = str(event.announcement_session)
        rows.append(
            f"({int(event.event_id)}, {int(event.secid)}, "
            f"'{announcement}'::date, '{cutoff}'::date, '{session}')")
    if not rows:
        raise ValueError("option event batch is empty")
    values = ",\n".join(rows)
    return f"""
        with events(event_id, secid, anndats_act, cutoff, announcement_session) as (
            values {values}
        ), entries as (
            select e.*, d.entry_date
            from events e
            cross join lateral (
                select max(q.date) as entry_date
                from optionm.opprcd{year} q
                where q.secid = e.secid
                  and q.date between e.cutoff - interval '7 days' and e.cutoff
            ) d
            where d.entry_date is not null
        ), pairs as (
            select e.event_id, e.secid, e.anndats_act,
                   e.announcement_session, e.entry_date, q.exdate,
                   q.strike_price / 1000.0 as strike, u.close as underlying_close,
                   max(q.best_bid) filter (where q.cp_flag = 'C') as call_bid,
                   max(q.best_offer) filter (where q.cp_flag = 'C') as call_ask,
                   max(q.impl_volatility) filter (where q.cp_flag = 'C') as call_iv,
                   max(q.delta) filter (where q.cp_flag = 'C') as call_delta,
                   max(q.open_interest) filter (where q.cp_flag = 'C') as call_oi,
                   max(q.volume) filter (where q.cp_flag = 'C') as call_volume,
                   max(q.optionid) filter (where q.cp_flag = 'C') as call_optionid,
                   max(q.best_bid) filter (where q.cp_flag = 'P') as put_bid,
                   max(q.best_offer) filter (where q.cp_flag = 'P') as put_ask,
                   max(q.impl_volatility) filter (where q.cp_flag = 'P') as put_iv,
                   max(q.delta) filter (where q.cp_flag = 'P') as put_delta,
                   max(q.open_interest) filter (where q.cp_flag = 'P') as put_oi,
                   max(q.volume) filter (where q.cp_flag = 'P') as put_volume,
                   max(q.optionid) filter (where q.cp_flag = 'P') as put_optionid
            from entries e
            join optionm.opprcd{year} q
              on q.secid = e.secid and q.date = e.entry_date
            join optionm.secprd{year} u
              on u.secid = e.secid and u.date = e.entry_date
            where q.exdate >= case when e.announcement_session = 'before_open'
                    then e.anndats_act else e.anndats_act + interval '1 day' end
              and q.exdate <= e.anndats_act + interval '{max_days} days'
              and q.ss_flag = '0' and q.best_bid >= 0
              and q.best_offer >= q.best_bid and q.best_offer > 0
              and q.impl_volatility > 0 and abs(q.delta) between 0.15 and 0.85
            group by e.event_id, e.secid, e.anndats_act,
                     e.announcement_session, e.entry_date, q.exdate,
                     q.strike_price, u.close
            having count(*) filter (where q.cp_flag = 'C') > 0
               and count(*) filter (where q.cp_flag = 'P') > 0
        ), ranked as (
            select p.*, row_number() over (
                partition by event_id, exdate
                order by abs(strike - underlying_close),
                         (call_ask-call_bid) + (put_ask-put_bid), strike
            ) as distance_rank
            from pairs p
        )
        select * from ranked where distance_rank <= 3
        order by event_id, exdate, distance_rank
    """


def download_option_straddles_for_events(
        events: pd.DataFrame, output: str | Path, env_file: str | Path = ".env",
        start_year: int = 2020, end_year: int = 2024, batch_size: int = 400,
        max_days: int = 28) -> pd.DataFrame:
    """Query raw OptionMetrics in indexed event batches and checkpoint yearly."""
    source = events.copy()
    source["anndats_act"] = pd.to_datetime(source["anndats_act"])
    source = source[source["announcement_session"].isin(
        ["before_open", "after_close"])]
    source = source[source["anndats_act"].dt.year.between(start_year, end_year)]
    source["option_quote_cutoff"] = source["anndats_act"] - pd.to_timedelta(
        source["announcement_session"].eq("before_open").astype(int), unit="D")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    connection = wrds_connection(env_file)
    try:
        for year in range(start_year, end_year + 1):
            annual = source[source["anndats_act"].dt.year.eq(year)]
            print(f"OptionMetrics {year}: {len(annual):,} events", flush=True)
            for start in range(0, len(annual), batch_size):
                batch = annual.iloc[start:start + batch_size]
                pieces.append(connection.raw_sql(
                    option_event_batch_sql(batch, year, max_days)))
                print(f"  {min(start + batch_size, len(annual)):,}/{len(annual):,}",
                      flush=True)
            checkpoint = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            checkpoint.to_parquet(output, index=False)
    finally:
        connection.close()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    for column in ("anndats_act", "entry_date", "exdate"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    frame.to_parquet(output, index=False)
    return frame


def option_calendar_exit_sql(calendars: pd.DataFrame, year: int) -> str:
    """Quote the long back straddle at the front expiration close."""
    rows = []
    for trade in calendars.itertuples(index=False):
        exit_date = pd.Timestamp(trade.exdate_front).date().isoformat()
        rows.append(
            f"({int(trade.event_id)}, {int(trade.secid_front)}, "
            f"'{exit_date}'::date, {float(trade.strike)}, "
            f"{int(trade.call_optionid_back)}, {int(trade.put_optionid_back)})")
    values = ",\n".join(rows)
    return f"""
        with trades(event_id, secid, exit_date, strike,
                    call_optionid, put_optionid) as (values {values})
        select t.event_id, t.exit_date, t.strike, u.close as exit_underlying,
               max(q.best_bid) filter (where q.optionid=t.call_optionid) as back_call_bid,
               max(q.best_offer) filter (where q.optionid=t.call_optionid) as back_call_ask,
               max(q.best_bid) filter (where q.optionid=t.put_optionid) as back_put_bid,
               max(q.best_offer) filter (where q.optionid=t.put_optionid) as back_put_ask
        from trades t
        join optionm.opprcd{year} q
          on q.date=t.exit_date
         and q.optionid in (t.call_optionid, t.put_optionid)
        join optionm.secprd{year} u on u.secid=t.secid and u.date=t.exit_date
        group by t.event_id, t.exit_date, t.strike, u.close
        having count(*) filter (where q.optionid=t.call_optionid)>0
           and count(*) filter (where q.optionid=t.put_optionid)>0
        order by t.event_id
    """


def download_calendar_exits(calendars: pd.DataFrame, output: str | Path,
                            env_file: str | Path = ".env",
                            batch_size: int = 500) -> pd.DataFrame:
    """Download back-leg marks at each selected front expiration."""
    source = calendars.copy()
    source["exdate_front"] = pd.to_datetime(source["exdate_front"])
    pieces = []
    connection = wrds_connection(env_file)
    try:
        for year, annual in source.groupby(source["exdate_front"].dt.year, sort=True):
            print(f"Calendar exits {year}: {len(annual):,}", flush=True)
            for start in range(0, len(annual), batch_size):
                batch = annual.iloc[start:start + batch_size]
                pieces.append(connection.raw_sql(option_calendar_exit_sql(batch, int(year))))
    finally:
        connection.close()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def option_calendar_close_sql(calendars: pd.DataFrame, year: int) -> str:
    """Quote all four calendar legs at the first post-announcement close."""
    rows = []
    for trade in calendars.itertuples(index=False):
        exit_date = pd.Timestamp(trade.reaction_date).date().isoformat()
        rows.append(
            f"({int(trade.event_id)}, {int(trade.secid_front)}, "
            f"'{exit_date}'::date, {int(trade.call_optionid_front)}, "
            f"{int(trade.put_optionid_front)}, {int(trade.call_optionid_back)}, "
            f"{int(trade.put_optionid_back)})")
    values = ",\n".join(rows)
    fields = []
    for leg in ("front_call", "front_put", "back_call", "back_put"):
        option = leg + "_optionid"
        fields.extend([
            f"max(q.best_bid) filter (where q.optionid=t.{option}) as {leg}_bid",
            f"max(q.best_offer) filter (where q.optionid=t.{option}) as {leg}_ask",
        ])
    return f"""
        with trades(event_id, secid, exit_date, front_call_optionid,
                    front_put_optionid, back_call_optionid,
                    back_put_optionid) as (values {values})
        select t.event_id, t.exit_date, u.close as exit_underlying,
               {', '.join(fields)}
        from trades t
        join optionm.opprcd{year} q
          on q.date=t.exit_date and q.optionid in (
             t.front_call_optionid, t.front_put_optionid,
             t.back_call_optionid, t.back_put_optionid)
        join optionm.secprd{year} u on u.secid=t.secid and u.date=t.exit_date
        group by t.event_id, t.exit_date, u.close
        having count(distinct q.optionid) = 4
        order by t.event_id
    """


def download_calendar_post_event_marks(
        calendars: pd.DataFrame, output: str | Path,
        env_file: str | Path = ".env", batch_size: int = 500) -> pd.DataFrame:
    """Download executable post-event bid/ask marks for all four legs."""
    source = calendars.copy()
    source["reaction_date"] = pd.to_datetime(source["reaction_date"])
    pieces = []
    connection = wrds_connection(env_file)
    try:
        for year, annual in source.groupby(source["reaction_date"].dt.year, sort=True):
            print(f"Calendar post-event marks {year}: {len(annual):,}", flush=True)
            for start in range(0, len(annual), batch_size):
                pieces.append(connection.raw_sql(option_calendar_close_sql(
                    annual.iloc[start:start + batch_size], int(year))))
    finally:
        connection.close()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def standardized_option_batch_sql(events: pd.DataFrame, year: int) -> str:
    """ATM standardized-option term structure for explicit event keys."""
    rows = []
    for event in events.itertuples(index=False):
        cutoff = pd.Timestamp(event.option_quote_cutoff).date().isoformat()
        rows.append(f"({int(event.event_id)}, {int(event.secid)}, '{cutoff}'::date)")
    return f"""
        with events(event_id, secid, cutoff) as (values {','.join(rows)}),
        entries as (
            select e.*, d.entry_date
            from events e cross join lateral (
                select max(s.date) as entry_date from optionm.stdopd{year} s
                where s.secid=e.secid
                  and s.date between e.cutoff-interval '7 days' and e.cutoff
            ) d where d.entry_date is not null
        )
        select e.event_id, e.secid, e.entry_date, s.days,
               max(s.forward_price) as forward_price,
               max(s.strike_price) as strike_price,
               max(s.premium) filter(where s.cp_flag='C') as call_premium,
               max(s.impl_volatility) filter(where s.cp_flag='C') as call_iv,
               max(s.delta) filter(where s.cp_flag='C') as call_delta,
               max(s.premium) filter(where s.cp_flag='P') as put_premium,
               max(s.impl_volatility) filter(where s.cp_flag='P') as put_iv,
               max(s.delta) filter(where s.cp_flag='P') as put_delta
        from entries e join optionm.stdopd{year} s
          on s.secid=e.secid and s.date=e.entry_date
        where s.days in (10,30,60,91)
        group by e.event_id,e.secid,e.entry_date,s.days
        having count(*) filter(where s.cp_flag='C')>0
           and count(*) filter(where s.cp_flag='P')>0
        order by e.event_id,s.days
    """


def download_standardized_option_terms(
        events: pd.DataFrame, output: str | Path, env_file: str | Path = ".env",
        start_year: int = 2020, end_year: int = 2024,
        batch_size: int = 1000) -> pd.DataFrame:
    """Download compact 10/30/60/91-day ATM standardized option surfaces."""
    source = events.copy()
    source["anndats_act"] = pd.to_datetime(source["anndats_act"])
    source = source[source["announcement_session"].isin(
        ["before_open", "after_close"])]
    source = source[source["anndats_act"].dt.year.between(start_year, end_year)]
    source["option_quote_cutoff"] = source["anndats_act"] - pd.to_timedelta(
        source["announcement_session"].eq("before_open").astype(int), unit="D")
    pieces = []
    connection = wrds_connection(env_file)
    try:
        for year in range(start_year, end_year + 1):
            annual = source[source["anndats_act"].dt.year.eq(year)]
            print(f"Standardized options {year}: {len(annual):,}", flush=True)
            for start in range(0, len(annual), batch_size):
                pieces.append(connection.raw_sql(standardized_option_batch_sql(
                    annual.iloc[start:start + batch_size], year)))
    finally:
        connection.close()
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def attach_dated_industry(events: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Attach only the classification effective on each announcement date."""
    left = events.copy()
    right = names.copy()
    left["anndats_act"] = pd.to_datetime(left["anndats_act"])
    right["namedt"] = pd.to_datetime(right["namedt"])
    right["nameendt"] = pd.to_datetime(right["nameendt"]).fillna(pd.Timestamp("2100-01-01"))
    joined = left.merge(right, on="permno", how="left")
    joined = joined[(joined["namedt"] <= joined["anndats_act"])
                    & (joined["anndats_act"] <= joined["nameendt"])]
    joined = (joined.sort_values(["event_id", "namedt"], kind="stable")
              .drop_duplicates("event_id", keep="last"))
    sic = pd.to_numeric(joined["siccd"], errors="coerce")
    naics = joined["naics"].astype("string")
    joined["sic2"] = (sic // 100).astype("Int64")
    joined["sic3"] = (sic // 10).astype("Int64")
    joined["naics2"] = pd.to_numeric(naics.str.slice(0, 2), errors="coerce").astype("Int64")
    joined["naics3"] = pd.to_numeric(naics.str.slice(0, 3), errors="coerce").astype("Int64")
    return joined.reset_index(drop=True)


def attach_crsp_reactions(events: pd.DataFrame, returns: pd.DataFrame,
                          max_horizon: int = 10) -> pd.DataFrame:
    """Attach close-to-close paths aligned to the release's trading session.

    After-close announcements react on the first trading day strictly after
    the release date. Before-open announcements react on the first trading day
    on or after it. Market-hours and unknown-time releases are omitted because
    daily CRSP data cannot separate pre- and post-release returns for them.
    """
    clean = events[events["announcement_session"].isin(
        ["before_open", "after_close"]
    )].copy()
    ret = returns[["permno", "date", "ret"]].copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["ret"] = pd.to_numeric(ret["ret"], errors="coerce")
    ret = ret.dropna(subset=["ret"]).sort_values(["permno", "date"])

    records = []
    grouped = {int(permno): group.reset_index(drop=True)
               for permno, group in ret.groupby("permno", sort=False)}
    for event in clean.itertuples(index=False):
        history = grouped.get(int(event.permno))
        if history is None:
            continue
        dates = history["date"].to_numpy(dtype="datetime64[ns]")
        announcement = np.datetime64(event.anndats_act)
        side = "right" if event.announcement_session == "after_close" else "left"
        start = int(np.searchsorted(dates, announcement, side=side))
        if start < 20 or start + max_horizon > len(history):
            continue
        forward = history["ret"].to_numpy(dtype=np.float64)[start:start + max_horizon]
        prior = history["ret"].to_numpy(dtype=np.float64)[start - 20:start]
        if not np.isfinite(forward).all() or not np.isfinite(prior).all():
            continue
        item = event._asdict()
        log_forward = np.log1p(np.clip(forward, -0.999, None))
        for horizon in (1, 2, 5, 10):
            if horizon <= max_horizon:
                item[f"reaction_{horizon}d"] = float(np.expm1(log_forward[:horizon].sum()))
        item["reaction_date"] = history.iloc[start]["date"]
        item["pre_20d_vol"] = float(np.std(prior, ddof=1) * np.sqrt(252))
        item["post_5d_vol"] = float(np.std(forward[:5], ddof=1) * np.sqrt(252))
        item["pre_5d_return"] = float(np.expm1(np.log1p(
            np.clip(prior[-5:], -0.999, None)).sum()))
        records.append(item)
    return pd.DataFrame(records)
