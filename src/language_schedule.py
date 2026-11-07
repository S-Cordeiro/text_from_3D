"""Paris night window used by Slurm submission and the training process."""
import argparse
import math
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

PARIS = ZoneInfo('Europe/Paris')


def night_window(now=None):
    now = (now or datetime.now(PARIS)).astimezone(PARIS)
    if now.hour < 8:
        begin = now
        end_date = now.date()
    elif now.hour < 18:
        begin = datetime.combine(now.date(), time(18), PARIS)
        end_date = now.date() + timedelta(days=1)
    else:
        begin = now
        end_date = now.date() + timedelta(days=1)
    return begin, datetime.combine(end_date, time(8), PARIS)


def night_seconds(now=None, end_epoch=None):
    now = (now or datetime.now(PARIS)).astimezone(PARIS)
    if 8 <= now.hour < 18:
        raise ValueError('Training is forbidden between 08:00 and 18:00 Europe/Paris')
    _, end = night_window(now)
    return math.floor(min(end.timestamp(), end_epoch or end.timestamp()) - now.timestamp())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['schedule', 'budget'])
    parser.add_argument('--end-epoch', type=float)
    args = parser.parse_args()
    if args.action == 'schedule':
        begin, end = night_window()
        print(begin.strftime('%Y-%m-%dT%H:%M:%S'), end.strftime('%Y-%m-%dT%H:%M:%S'), int(end.timestamp()))
    else:
        try:
            seconds = night_seconds(end_epoch=args.end_epoch)
        except ValueError as exc:
            parser.error(str(exc))
        if seconds < 600:
            parser.error('Less than ten minutes remain before the night deadline')
        print(seconds)


if __name__ == '__main__':
    main()
