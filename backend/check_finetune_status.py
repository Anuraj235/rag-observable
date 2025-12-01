from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def format_ts(ts):
    """Convert a timestamp (int/float or ISO string) to 'YYYY-MM-DD HH:MM:SS UTC'."""
    if not ts:
        return "n/a"

    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        # API often returns an int seconds-since-epoch, but this is safe if it ever sends ISO.
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))

    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    print("=== Most relevant fine-tuning job ===")

    # Just grab the single most recent job
    jobs = client.fine_tuning.jobs.list(limit=1)
    if not jobs.data:
        print("No fine-tuning jobs found.")
        return

    job = jobs.data[0]

    print(f"id: {job.id}")
    print(f"status: {job.status}")
    print(f"base model: {job.model}")
    print(f"fine-tuned: {getattr(job, 'fine_tuned_model', None) or 'None'}")
    print(f"training_file: {getattr(job, 'training_file', 'n/a')}")
    print(f"validation_file: {getattr(job, 'validation_file', 'n/a')}")
    print(f"error: {job.error}")

    created = getattr(job, "created_at", None)
    finished = getattr(job, "finished_at", None)

    print(f"created_at: {format_ts(created)}")
    print(f"finished_at: {format_ts(finished)}")

    # Show elapsed / duration
    if created is not None:
        now = datetime.now(timezone.utc).timestamp()
        end = finished if finished is not None else now
        elapsed_min = (end - created) / 60.0

        if finished is None:
            print(f"elapsed: {elapsed_min:.1f} minutes (still running)")
        else:
            print(f"duration: {elapsed_min:.1f} minutes (completed)")

    print("\n=== Recent events for this job ===")
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job.id,
        limit=5
    )

    for e in reversed(events.data):
        ts = format_ts(e.created_at)
        level = getattr(e, "level", "info").upper()
        print(f"[{ts}] {level}: {e.message}")


if __name__ == "__main__":
    main()
 