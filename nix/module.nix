self: { config, lib, pkgs, ... }:
let
  cfg = config.services.qwen-flash-trellis;
  package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
in {
  options.services.qwen-flash-trellis = {
    enable = lib.mkEnableOption "Qwen Flash trellis PLE-SSD serving";
    modelPath = lib.mkOption { type = lib.types.str; description = "Full downloaded checkpoint on a local SSD; readable by the service user."; };
    stateDirectory = lib.mkOption { type = lib.types.str; default = "/var/lib/qwen-flash-trellis"; description = "Writable runtime/cache directory; allow space for CUDA wheels and compilation."; };
    user = lib.mkOption { type = lib.types.str; default = "qwen-trellis"; };
    apiKeyFile = lib.mkOption { type = lib.types.nullOr lib.types.str; default = null; description = "Runtime credential file, never copied into the Nix store."; };
    host = lib.mkOption { type = lib.types.str; default = "127.0.0.1"; };
    port = lib.mkOption { type = lib.types.port; default = 8099; };
    kvGiB = lib.mkOption { type = lib.types.number; default = 14.6; description = "KV pool in GiB; includes fractional budgets."; };
    device = lib.mkOption { type = lib.types.str; default = "0"; };
  };
  config = lib.mkIf cfg.enable {
    users.groups.qwen-trellis = {};
    users.users.qwen-trellis = { isSystemUser = true; group = "qwen-trellis"; };
    systemd.tmpfiles.rules = [ "d ${cfg.stateDirectory} 0700 ${cfg.user} qwen-trellis -" ];
    systemd.services.qwen-flash-trellis = {
      description = "Qwen Flash trellis inference (no GPU tuning)";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      unitConfig.RequiresMountsFor = [ cfg.modelPath cfg.stateDirectory ];
      environment = { T3_WORK_DIR = cfg.stateDirectory; HOME = cfg.stateDirectory; CUDA_VISIBLE_DEVICES = cfg.device; };
      serviceConfig = {
        User = cfg.user;
        Group = "qwen-trellis";
        SupplementaryGroups = [ "video" "render" ];
        WorkingDirectory = cfg.stateDirectory;
        ExecStartPre = "${package}/bin/t3 setup";
        ExecStart = lib.escapeShellArgs [ "${package}/bin/t3" "serve" "--model" cfg.modelPath "--host" cfg.host "--port" (toString cfg.port) "--kv-gib" (toString cfg.kvGiB) ];
        LoadCredential = lib.optional (cfg.apiKeyFile != null) "api-key:${cfg.apiKeyFile}";
        Restart = "on-failure";
        RestartSec = 30;
        TimeoutStartSec = "2h";
        TimeoutStopSec = 120;
      };
    };
  };
}
