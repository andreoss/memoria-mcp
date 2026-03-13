import { describe, expect, test } from "bun:test";
import { parseProjectFromRemote } from "./project";

describe("parseProjectFromRemote", () => {
  test("https remote with a .git suffix", () => {
    expect(parseProjectFromRemote("https://github.com/example-org/example-repo.git")).toBe("example-org-example-repo");
  });

  test("https remote without a .git suffix", () => {
    expect(parseProjectFromRemote("https://github.com/acme/widgets")).toBe("acme-widgets");
  });

  test("standard scp-style ssh remote", () => {
    expect(parseProjectFromRemote("git@github.com:openai/gym.git")).toBe("openai-gym");
  });

  test("ssh remote with a custom host alias", () => {
    expect(parseProjectFromRemote("git@github.com-work:acme/widgets.git")).toBe("acme-widgets");
  });

  test("trailing slash is ignored", () => {
    expect(parseProjectFromRemote("https://github.com/acme/widgets/")).toBe("acme-widgets");
  });

  test("leading and trailing whitespace is ignored", () => {
    expect(parseProjectFromRemote("  https://github.com/acme/widgets.git\n")).toBe("acme-widgets");
  });

  test("returns null when no owner/repo can be parsed", () => {
    expect(parseProjectFromRemote("not-a-remote")).toBeNull();
  });

  test("returns null for an empty string", () => {
    expect(parseProjectFromRemote("")).toBeNull();
  });
});
