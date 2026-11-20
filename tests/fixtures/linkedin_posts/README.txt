LinkedIn posts fixtures (socmint/collect/linkedin_posts.py)

Captured live 2026-11-19 ~12:52 UTC from linkedin.com's own content search
(keywords "bank of america" in quotes, sortBy date_posted, datePosted past-24h) in a
headless Chrome with the owner's session. Collected at 2026-11-19T12:52:25Z (tests use it).

live_search_page.html       the search page document, reduced to a minimal page whose
                            window.__como_rehydration__ holds the trimmed flight stream
                            (3 results: activities 7529163328582156288,
                            7529162820656082944, 7529162861017776128)
live_search_pagination.rsc  the first contentSearchResults pagination response, trimmed
                            (3 results: 7529162869180129280, 7529162428027248641,
                            7529162074019622913 = a group post)

How they were trimmed (originals 3.4 MB / 2.6 MB):
  - only the rows reachable from each "feed-full-update" node are kept, copied verbatim
    except for pruned subtrees; synthetic rows 0-3 hold the item list, the modelStates
    entries keyed by the items' URNs, and the next-page PaginationRequest (as a dict in
    the page, as a JSON string in the pagination stream: both forms were seen live);
  - pruned: comment boxes / comment lists, follow + group-join buttons, reaction menus,
    GIF pickers, SetState / ShowToast actions, CSS class names, styles, tracking buffers;
  - account data removed: every row of the account's own comment box (its avatar,
    "commentBox-*" state), memberFirstName / memberLastName / memberHeadline objects
    (the comment box of the logged-in account), viewerProfileId values replaced by
    "[viewer-redacted]", the whole nav bar / "Me" menu / messaging / notifications
    (outside the result items, never copied);
  - scanned after trimming: no cookie value, no account name / vanity / profile id, and
    the account's avatar asset id does not occur.
The parser yields identical items from the trimmed and the original captures.

Voyager provider fixtures (socmint/collect/linkedin_voyager.py)

Captured live 2026-11-19 ~13:56:40 UTC (tests use 13:56:40Z as the collection time) by
two direct GETs to /voyager/api/graphql (queryId voyagerSearchDashClusters.5ba3...855c,
count 10, keywords "bank of america" in quotes) with the owner's session:

live_voyager_search_relevance.json    queryParameters resultType=CONTENT only (relevance
                                      order, posts from 1 d to 3 w old)
live_voyager_search_date_sorted.json  datePosted=past-week, resultType=CONTENT,
                                      sortBy=date_posted (newest first, paging.total 245)

How they were trimmed:
  - data.data.searchDashClustersByAll keeps paging, elements and metadata.totalResultCount;
  - included keeps Update (metadata.backendUrn/shareUrn, entityUrn, actor, commentary,
    content, resharedUpdate, *socialDetail, socialContent), SocialDetail,
    SocialActivityCounts, Profile, Company and SocialPermissions; $recipeTypes and
    null / empty fields dropped;
  - the viewer's own member id (in SocialPermissions urns, SocialDetail.*socialPermissions
    and the rcm= parameter of each shareUrl) is replaced by VIEWER_PLACEHOLDER; the cookie
    values were scrubbed before the captures were saved. A structural test checks that the
    placeholder is the only viewer id and that every other profile id belongs to an author
    or a mentioned profile.
