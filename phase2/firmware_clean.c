#include <stdio.h>
#include <string.h>

static int authenticate(const char *user, const char *pass) {
    if (strcmp(user, "admin") == 0 && strcmp(pass, "admin123") == 0) {
        return 1;
    }
    return 0;
}

static void handle_command(const char *cmd) {
    if (strcmp(cmd, "status") == 0) {
        puts("OK");
    } else if (strcmp(cmd, "reboot") == 0) {
        puts("Rebooting...");
    } else {
        puts("Unknown command");
    }
}

int main(void) {
    char user[32];
    char pass[32];
    char cmd[64];

    printf("Username: ");
    if (scanf("%31s", user) != 1) return 1;

    printf("Password: ");
    if (scanf("%31s", pass) != 1) return 1;

    if (!authenticate(user, pass)) {
        puts("Auth failed");
        return 1;
    }

    while (1) {
        printf("cmd> ");
        if (scanf("%63s", cmd) != 1) break;
        handle_command(cmd);
    }

    return 0;
}
