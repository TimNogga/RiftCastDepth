import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.run("pytest", "tests", external=True)


@nox.session
def benchmarks(session: nox.Session) -> None:
    """Runs the benchmarks."""
    session.run("pytest", "benchmarks", "--benchmark-sort=fullname", external=True)
