# Publication follow-up checklist

This is an evidence checklist, not a certification of compliance.
Do not mark an item complete merely because a disclaimer was added.

## External teacher rights — owner/provider action

- [ ] Identify the provider and exact actor/version used.
- [ ] Obtain written scope for rollout/label generation and student distillation.
- [ ] Confirm whether source, performance tables, videos, and derived checkpoints may be published.
- [ ] Confirm research/commercial use, required attribution, and confidentiality restrictions.
- [ ] Check whether work produced during the laboratory internship needs laboratory approval.
- [ ] Keep confidential correspondence outside Git; record a non-confidential decision in
      [EXTERNAL_TEACHER_RIGHTS.md](EXTERNAL_TEACHER_RIGHTS.md).

Until confirmed, keep affected claims provisional. Permission to access an actor
does not establish permission for all subsequent uses.

## Experiment evidence — experiment owner action

- [ ] Retain raw evaluation logs and manifests linked to the reported checkpoint hashes.
- [ ] Record initial-state selection, seeds, episode/window duration, termination criteria,
      and denominators; retain the failed cases.
- [ ] Keep the four-second 106/108 result separate from the three-second 108/108 result.
- [ ] Identify precisely which code, experiments, and debugging were personally performed.
- [ ] Only publish artifacts approved for disclosure. Do not upload restricted assets merely
      to make reproduction easier.
- [ ] For a publicly reproducible example, use a teacher and dataset with verified permissions.

See [RESULTS.md](RESULTS.md) for the current evidence boundary. Lack of public
artifacts is not evidence of fabrication, but the tables are not independently
verified by this repository.

## Previously published objects — repository owner / hosting-provider action

The current main branch was recreated as a clean snapshot. During the
2026-10-14 review, the two earlier public commits were still retrievable by
their old identifiers even though they were no longer main's ancestors.
Do not interpret a short commit list as proof of server-side erasure.

- [ ] Review advertised branches/tags, pull-request references, releases, and repository forks.
- [ ] Privately record affected old object identifiers and file paths; do not reprint
      sensitive values in public issues, tests, or documentation.
- [ ] Assess whether the exposed host/account information is still operationally sensitive
      and whether access controls need adjustment.
- [ ] Consult [GitHub's sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
      Eligible cached-object removal requires GitHub Support; ordinary repository writes
      cannot guarantee this. GitHub does not promise removal of non-sensitive content.
- [ ] Separately resolve the rights of any previously distributed restricted code.
      Server-side deletion does not retroactively establish permission, and clones may survive.

Do not file a copyright takedown on someone else's behalf without authority.
Do not repeatedly rewrite main, delete the repository, or change visibility
under the assumption that this guarantees erasure.

## Release verification — maintainer action

- [ ] Run `python -B tools/test_check_release.py`.
- [ ] Run `python -B tools/verify_a3_source.py` and `python -B tools/check_release.py`.
- [ ] Review staged content and reachable history separately; the checker scans working-tree files.
- [ ] Verify component licenses, source attribution, and personal contribution boundaries.
- [ ] Reassess the résumé entry after the rights and evidence items above are resolved.
