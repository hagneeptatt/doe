# Hosting the app for your institute

This guide is for whoever looks after a shared copy of the app. People using it only need [the manual](manual.md).

## Before you host it

**Decide where unpublished data may go.** People will upload experimental results to the app, and those results pass through whichever server hosts it. The app doesn't save them: they stay in memory for the browser session and are gone when it ends. They're still on that server while in use, so check your institute's data policy. An institute-run server is the safest default.

**This repository is public.** Git ignores `data/` and `archive/private/`, so run sheets and private data aren't uploaded by accident. Don't add unpublished results to a public copy of this repository.

**The app has no login.** Anyone who can reach its address can use it. Put it behind your institute's network, VPN or an authenticating proxy.

## Option 1: a container on an institute server or research platform (recommended)

This keeps data on institute infrastructure.

```bash
docker build -t doekit .
docker run -p 8501:8501 doekit
```

Then open <http://localhost:8501>. The health check is at `/_stcore/health`.

The image runs as a non-root user and doesn't need to write anywhere except `/tmp`, so it suits platforms that assign arbitrary user IDs, such as OpenShift. To deploy there, push the image to a registry your platform can read, then create an application from it with port 8501 exposed.

## Option 2: run it on a shared lab computer

```bash
conda env create -f environment.yml
conda activate doe
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Colleagues on the same network can then open `http://<computer-name>:8501`. It stops when the computer does, so this suits a small group. To run it as a service, use your operating system's service manager.

## Option 3: a public cloud service

Streamlit Community Cloud and similar services can deploy directly from a Git repository, and they're free to start with. `environment.yml` and `pyproject.toml` provide what they need, with `app/streamlit_app.py` as the entry point.

Only use this once you've checked the points above, because people's results pass through a third party's servers. Check the service's current terms. This route hasn't been tested with doekit.

## Keeping it working

- **After any change**, run `pytest` before redeploying.
- **Resources.** Each design or search runs on the server's processor while someone waits, usually for a few seconds. One or two CPUs and 2 GB of memory are a reasonable starting point for a small group.
- **Very large requests** are refused with an explanation, rather than left to run indefinitely. The limits are in `src/doekit/designs/`.
- **Updates.** Dependency versions are limited in `pyproject.toml`, and major version bumps should be tested before they're allowed.
